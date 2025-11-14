"""
Model training for Economedia PTS.

Key behaviors preserved:
- Run_id provided by entrypoint; unified across cv_build and training.
- One model per label (cap_90d, dne_90d, cap_30d, dne_30d).
- Feature prep.
- Bayesian Optimization over XGBoost hyperparameters (maximize mean ROC AUC across folds).
- Isotonic calibration on concatenated validation sets, weighted by natural counts.
- Global threshold chosen to maximize EXPECTED F1 on validation.
- Metrics computed for observed and expected distributions.

This module assembles outputs so that artifact saving & BQ persistence can happen
in later steps (artifact_io.py, metrics_to_bq.py), without changing training math.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from google.cloud import bigquery

from app.common.io import get_bq_client, gcs_upload_bytes, gcs_upload_text, gcs_upload_json
import io
from app.common.preprocessing import prepare_xy_compat
from app.common.utils import get_logger

# Optional: will be implemented in later steps; guarded calls below
try:
    from app.training import artifact_io
except Exception:  # pragma: no cover
    artifact_io = None

try:
    from app.training import metrics_to_bq
except Exception:  # pragma: no cover
    metrics_to_bq = None


# -----------------------
# Defaults from env (overridden by entrypoint params)
# -----------------------
PROJECT_ID = os.getenv("PROJECT_ID", "propensity-to-subscr-eng-prod")
BQ_LOCATION = os.getenv("BQ_LOCATION", "europe-west3")
BQ_DATASET = os.getenv("BQ_DATASET", "propensity_to_subscribe")

TABLE_DATA = f"{PROJECT_ID}.{BQ_DATASET}.train_data"
TABLE_META = f"{PROJECT_ID}.{BQ_DATASET}.cv_build_metadata"
TABLE_PREP = f"{PROJECT_ID}.{BQ_DATASET}.prep_metadata"

RANDOM_STATE = 42
N_BAYES_INIT = 5
N_BAYES_ITER = 55


# -----------------------
# Helpers reading from BQ
# -----------------------


def load_train_table(client: bigquery.Client, run_id: str, label_tag: str) -> pd.DataFrame:
    sql = f"""
    SELECT *
    FROM `{TABLE_DATA}`
    WHERE run_id = @run_id AND label = @label
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
                bigquery.ScalarQueryParameter("label", "STRING", label_tag),
            ]
        ),
    )
    df = job.result().to_dataframe()
    # types are already normalized by cv_build.py
    return df


def load_meta_counts(client: bigquery.Client, run_id: str, label_tag: str) -> pd.DataFrame:
    sql = f"""
    SELECT split, CAST(fold AS INT64) AS fold, n, pos, neg
    FROM `{TABLE_META}`
    WHERE run_id = @run_id AND label_tag = @label AND split IN ('train','val','test')
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
                bigquery.ScalarQueryParameter("label", "STRING", label_tag),
            ]
        ),
    )
    return job.result().to_dataframe()


# -----------------------
# Metrics / expected counts
# -----------------------

def class_weights_from_counts(y_true: np.ndarray | pd.Series, pos_true: int, neg_true: int) -> np.ndarray:
    """Weights so the weighted positives/negatives match natural counts."""
    arr = np.asarray(y_true)
    pos_obs = max(int((arr == 1).sum()), 1)
    neg_obs = max(int((arr == 0).sum()), 1)
    w_pos = pos_true / pos_obs
    w_neg = neg_true / neg_obs
    return np.where(arr == 1, w_pos, w_neg).astype("float64")


def _clip01(a: np.ndarray) -> np.ndarray:
    """Ensure probability array is finite and in [0, 1]."""
    a = np.asarray(a, dtype=float)
    a = np.nan_to_num(a, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(a, 0.0, 1.0)

def best_expected_f1_threshold_global(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pos_true: int,
    neg_true: int,
    grid: int = 400,
) -> Tuple[float, float]:
    """
    Scan thresholds; compute sample TPR/FPR; scale to expected (pos_true, neg_true);
    return threshold that maximizes expected F1.
    """
    ts = np.linspace(0.0, 1.0, grid)
    best_t, best_val = 0.5, -1.0
    for t in ts:
        pred = (y_prob >= t).astype(int)
        tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        pos_s = max(1, int((y_true == 1).sum()))
        neg_s = max(1, int((y_true == 0).sum()))
        tpr = tp_s / pos_s
        fpr = fp_s / neg_s
        tp = tpr * pos_true
        fp = fpr * neg_true
        fn = (1 - tpr) * pos_true
        prec = tp / max(tp + fp, 1e-12)
        rec = tp / max(tp + fn, 1e-12)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
        if f1 > best_val:
            best_val, best_t = f1, t
    return float(best_t), float(best_val)


# -----------------------
# Metrics helpers & visualization
# -----------------------

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    pos = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    neg = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return float((pos - neg).abs().max())


def weighted_logloss(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray) -> float:
    return float(log_loss(y_true, y_prob, sample_weight=w, labels=[0, 1]))


def weighted_brier(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray) -> float:
    return float(np.average((y_true - y_prob) ** 2, weights=w))


def expected_counts_from_rates(tpr: float, fpr: float, pos_true: int, neg_true: int) -> Tuple[float, float, float, float]:
    tp = tpr * pos_true
    fn = (1.0 - tpr) * pos_true
    fp = fpr * neg_true
    tn = (1.0 - fpr) * neg_true
    return tp, fp, tn, fn


def expected_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thr: float,
    pos_true: int,
    neg_true: int,
) -> Dict[str, float]:
    """Expected confusion & derived metrics at ``thr`` for true prevalence."""

    pred = (y_prob >= thr).astype(int)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    pos_s = max(1, int((y_true == 1).sum()))
    neg_s = max(1, int((y_true == 0).sum()))
    tpr = tp_s / pos_s
    fpr = fp_s / neg_s

    tp, fp, tn, fn = expected_counts_from_rates(tpr, fpr, pos_true, neg_true)
    n_true = pos_true + neg_true
    prec = tp / max(tp + fp, 1e-12)
    rec = tp / max(tp + fn, 1e-12)
    acc = (tp + tn) / max(n_true, 1e-12)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-12)

    return {
        "exp_tp": float(tp),
        "exp_fp": float(fp),
        "exp_tn": float(tn),
        "exp_fn": float(fn),
        "exp_precision": float(prec),
        "exp_recall": float(rec),
        "exp_accuracy": float(acc),
        "exp_f1": float(f1),
        "tpr": float(tpr),
        "fpr": float(fpr),
    }


def prob_metrics(y_true: np.ndarray, y_prob: np.ndarray, w: Optional[np.ndarray] = None) -> Dict[str, float]:
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
    if w is None:
        pr_auc = float(average_precision_score(y_true, y_prob))
        ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
        br = float(brier_score_loss(y_true, y_prob))
    else:
        pr_auc = float(average_precision_score(y_true, y_prob, sample_weight=w))
        ll = weighted_logloss(y_true, y_prob, w)
        br = weighted_brier(y_true, y_prob, w)
    return {"auc": auc, "pr_auc": pr_auc, "logloss": ll, "brier": br}


def random_baseline_expected_metrics(rate_pred: float, pos_true: int, neg_true: int) -> Dict[str, float]:
    tp = rate_pred * pos_true
    fp = rate_pred * neg_true
    fn = (1 - rate_pred) * pos_true
    tn = (1 - rate_pred) * neg_true
    n = pos_true + neg_true
    prec = tp / max(tp + fp, 1e-12)
    rec = tp / max(tp + fn, 1e-12)
    acc = (tp + tn) / max(n, 1e-12)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": float(prec),
        "recall": float(rec),
        "accuracy": float(acc),
        "f1": float(f1),
    }

def _fig_to_png_bytes() -> bytes:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.getvalue()



def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, thr: float, out_gcs_uri: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    tpr_point = tp / max(tp + fn, 1e-12)
    fpr_point = fp / max(fp + tn, 1e-12)

    plt.figure()
    plt.plot(fpr, tpr, label="Model ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.scatter([fpr_point], [tpr_point], marker="o", label=f"Thr={thr:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (test)")
    plt.legend()
    png = _fig_to_png_bytes()
    gcs_upload_bytes(out_gcs_uri, png, content_type="image/png")






def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, thr: float, out_gcs_uri: str):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    prevalence = float(np.mean(y_true)) if len(y_true) else 0.0
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec_point = tp / max(tp + fp, 1e-12)
    rec_point = tp / max(tp + fn, 1e-12)

    plt.figure()
    plt.plot(rec, prec, label="Model PR")
    plt.hlines(prevalence, 0, 1, linestyles="--", label=f"Baseline (prev={prevalence:.3%})")
    plt.scatter([rec_point], [prec_point], marker="o", label=f"Thr={thr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (test)")
    plt.legend()
    png = _fig_to_png_bytes()
    gcs_upload_bytes(out_gcs_uri, png, content_type="image/png")


def plot_calibration(calib_df: pd.DataFrame, mean_pred_col: str, rate_col: str, out_gcs_uri: str, title: str):
    plt.figure()
    x = calib_df[mean_pred_col].values
    y = calib_df[rate_col].values
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.scatter(x, y, label="Bins")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed rate")
    plt.title(title)
    plt.legend()
    png = _fig_to_png_bytes()
    gcs_upload_bytes(out_gcs_uri, png, content_type="image/png")


def plot_lift(tbl: pd.DataFrame, x_col: str, y_col: str, out_gcs_uri: str, title: str):
    plt.figure()
    plt.plot(tbl[x_col], tbl[y_col], marker="o", label="Model")
    plt.hlines(1.0, tbl[x_col].min(), tbl[x_col].max(), linestyles="--", label="Baseline")
    plt.xlabel("Decile (10=top)")
    plt.ylabel("Lift")
    plt.title(title)
    plt.legend()
    png = _fig_to_png_bytes()
    gcs_upload_bytes(out_gcs_uri, png, content_type="image/png")


def _annotate_cm(ax, cm: np.ndarray):
    ax.imshow(cm, interpolation="nearest")
    for (i, j), v in np.ndenumerate(cm):
        if float(v).is_integer():
            txt = f"{int(v)}"
        else:
            txt = f"{v:.1f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])


def plot_confusions_side_by_side(cm_left: np.ndarray, cm_right: np.ndarray, titles: Tuple[str, str], out_gcs_uri: str, suptitle: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    _annotate_cm(axes[0], cm_left)
    axes[0].set_title(titles[0])
    _annotate_cm(axes[1], cm_right)
    axes[1].set_title(titles[1])
    fig.suptitle(suptitle)
    png = _fig_to_png_bytes()
    gcs_upload_bytes(out_gcs_uri, png, content_type="image/png")


def plot_score_hist(y_true: np.ndarray, y_prob: np.ndarray, out_gcs_uri: str, bins: int = 40):
    plt.figure()
    plt.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, label="Negatives")
    plt.hist(y_prob[y_true == 1], bins=bins, alpha=0.6, label="Positives")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Score distribution by class (test)")
    plt.legend()
    png = _fig_to_png_bytes()
    gcs_upload_bytes(out_gcs_uri, png, content_type="image/png")


def plot_feature_importance(booster, out_gcs_uri: str, topn: int = 20):
    fmap = booster.get_score(importance_type="gain") or booster.get_score(importance_type="weight")
    items = sorted(fmap.items(), key=lambda kv: kv[1], reverse=True)[:topn]
    if not items:
        return
    labels, vals = zip(*items)
    plt.figure()
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {topn} Feature Importances")
    png = _fig_to_png_bytes()
    gcs_upload_bytes(out_gcs_uri, png, content_type="image/png")


def plot_metric_bars(model_metrics: Dict[str, float], baseline_metrics: Dict[str, float], title: str, out_gcs_uri: str):
    metrics = ["precision", "recall", "accuracy", "f1"]
    model_vals = [float(model_metrics.get(m, np.nan)) for m in metrics]
    base_vals = [float(baseline_metrics.get(m, np.nan)) for m in metrics]
    x = np.arange(len(metrics))
    width = 0.38

    plt.figure()
    bars_model = plt.bar(x - width / 2, model_vals, width=width, label="Model")
    bars_base = plt.bar(x + width / 2, base_vals, width=width, label="Random baseline")

    def _autolabel(bars):
        for b in bars:
            h = b.get_height()
            if np.isnan(h):
                label = "NA"
                h_plot = 0.0
            else:
                label = f"{h:.2f}"
                h_plot = h
            plt.text(b.get_x() + b.get_width() / 2, h_plot + 0.01, label, ha="center", va="bottom", fontsize=9)

    _autolabel(bars_model)
    _autolabel(bars_base)

    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    png = _fig_to_png_bytes()
    gcs_upload_bytes(out_gcs_uri, png, content_type="image/png")





# -----------------------
# Trainer
# -----------------------

@dataclass
class TrainOutputs:
    label: str
    best_params: Dict[str, Any]
    threshold_expected: float
    auc_val_mean: float
    artifact_gcs_prefix: str  # gs://.../runs/<run_id>/<label>

class LabelTrainer:
    def __init__(
        self,
        label_tag: str,
        *,
        run_id: str,
        gcs_model_bucket: str,
        vertex_model_display_name: str,
        vertex_model_registry_label: str,
    ):
        self.label_tag = label_tag
        self.target_col = "y"

        self.logger = get_logger(f"pts.training.{label_tag}")
        self.client = get_bq_client(project=PROJECT_ID, location=BQ_LOCATION)

        # Use the provided run_id (unified with cv_build), not "latest".
        self.run_id = run_id

        # Where ALL artifacts for this label will be uploaded
        self.gcs_prefix = f"{gcs_model_bucket.rstrip('/')}/runs/{run_id}/{label_tag}"

        # For registry/manifest context
        self.vertex_model_display_name = vertex_model_display_name
        self.vertex_model_registry_label = vertex_model_registry_label

        self.folds: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        self.test_df: Optional[pd.DataFrame] = None
        self.meta_counts: Optional[pd.DataFrame] = None

        self.best_params: Optional[Dict[str, Any]] = None
        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.val_thr_expected: Optional[float] = None
        self.metrics_df: Optional[pd.DataFrame] = None

    # ---------- load data ----------

    def load(self) -> None:
        df = load_train_table(self.client, self.run_id, self.label_tag)
        # Partition by split/fold (built by cv_build.py)
        self.test_df = df[df["split"] == "test"].copy()
        self.folds = {}
        for k in sorted(df.loc[df["split"].isin(["train", "val"]), "fold"].dropna().unique()):
            tr = df[(df["split"] == "train") & (df["fold"] == k)].copy()
            va = df[(df["split"] == "val") & (df["fold"] == k)].copy()
            if not tr.empty and not va.empty:
                self.folds[int(k)] = (tr, va)
        if not self.folds:
            raise RuntimeError(f"No train/val folds found for {self.label_tag}")

        # Natural counts from metadata
        self.meta_counts = load_meta_counts(self.client, self.run_id, self.label_tag)
        self.logger.info(
            "Loaded training data for label=%s run_id=%s | folds=%s | test_rows=%s",
            self.label_tag,
            self.run_id,
            len(self.folds),
            len(self.test_df) if self.test_df is not None else 0,
        )

    # ---------- bayesian optimization ----------

    def _cv_score(self, max_depth, gamma, colsample_bytree, learning_rate, n_estimators, reg_alpha, reg_lambda) -> float:
        """Objective to maximize: mean ROC AUC across folds."""
        params = dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=int(round(n_estimators)),
            max_depth=int(round(max_depth)),
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            colsample_bytree=float(colsample_bytree),
            reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda),
            subsample=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        aucs = []
        for k, (tr_df, va_df) in self.folds.items():
            X_tr, y_tr = prepare_xy_compat(tr_df, self.target_col)
            X_va, y_va = prepare_xy_compat(va_df, self.target_col)
            spw = self._scale_pos_weight(y_tr)
            model = xgb.XGBClassifier(**params, scale_pos_weight=spw)
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, p))
        return float(np.mean(aucs))

    def bayes_optimize(self) -> None:
        optimizer = BayesianOptimization(
            f=self._cv_score,
            pbounds={
                "max_depth": (3, 10),
                "gamma": (0.0, 1.5),
                "colsample_bytree": (0.6, 1.0),
                "learning_rate": (0.01, 0.2),
                "n_estimators": (300, 1000),
                "reg_alpha": (0.0, 1.0),
                "reg_lambda": (0.0, 1.5),
            },
            random_state=RANDOM_STATE,
            verbose=2,
        )
        optimizer.maximize(init_points=N_BAYES_INIT, n_iter=N_BAYES_ITER)
        self.best_params = dict(optimizer.max["params"])
        self.best_params["max_depth"] = int(round(self.best_params["max_depth"]))
        self.best_params["n_estimators"] = int(round(self.best_params["n_estimators"]))
        
        # Upload best_params.json directly to GCS
        best_params_uri = f"{self.gcs_prefix}/best_params.json"
        gcs_upload_json(best_params_uri, self.best_params, indent=2)


    # ---------- final fit + calibration + threshold ----------

    def fit_final(self) -> None:
        # Train on ALL train folds concatenated
        X_list, y_list = [], []
        for _, (tr_df, _) in self.folds.items():
            X_tr, y_tr = prepare_xy_compat(tr_df, self.target_col)
            X_list.append(X_tr)
            y_list.append(y_tr)
        X_all = pd.concat(X_list, axis=0)
        y_all = pd.concat(y_list, axis=0)
        spw = self._scale_pos_weight(y_all)

        params = dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            subsample=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **(self.best_params or {}),
        )
        self.model = xgb.XGBClassifier(**params, scale_pos_weight=spw)
        self.model.fit(X_all, y_all)

        # Upload model bytes (use raw format for in-memory streaming)
        booster = self.model.get_booster()
        model_bytes = booster.save_raw()  # binary (UBJ)
        gcs_upload_bytes(f"{self.gcs_prefix}/model_{self.label_tag}.ubj", model_bytes, content_type="application/octet-stream")
        


        # Validation concat for calibration & threshold selection
        Xv_list, yv_list = [], []
        for _, (_, va_df) in self.folds.items():
            X_va, y_va = prepare_xy_compat(va_df, self.target_col)
            Xv_list.append(X_va)
            yv_list.append(y_va)
        X_val_all = pd.concat(Xv_list, axis=0)
        y_val_all = pd.concat(yv_list, axis=0)
        p_val_all = self.model.predict_proba(X_val_all)[:, 1]

        # Expected weighting from natural counts (across all folds)
        pos_true_total, neg_true_total = self._natural_totals(split="val")
        w_val = class_weights_from_counts(y_val_all, pos_true_total, neg_true_total)

        # Isotonic calibration (validation-based)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_val_all, y_val_all, sample_weight=w_val)
        self.calibrator = iso
        # Upload calibrator as bytes
        try:
            import joblib
            buf = io.BytesIO()
            joblib.dump(iso, buf)
            buf.seek(0)
            gcs_upload_bytes(f"{self.gcs_prefix}/isotonic_calibrator_{self.label_tag}.joblib", buf.getvalue(), content_type="application/octet-stream")
        except Exception:
            pass

        # Calibrated probabilities for threshold selection
        p_val_cal = _clip01(iso.transform(p_val_all))

        # Choose threshold maximizing EXPECTED F1 on validation
        thr, best_exp_f1 = best_expected_f1_threshold_global(
            y_true=y_val_all.values,
            y_prob=p_val_cal,
            pos_true=pos_true_total,
            neg_true=neg_true_total,
            grid=400,
        )
        self.val_thr_expected = float(thr)
        # Upload chosen threshold
        thr_text = f"{self.val_thr_expected:.6f}\n"
        gcs_upload_text(f"{self.gcs_prefix}/threshold_expected_{self.label_tag}.txt", thr_text, content_type="text/plain")




    # ---------- evaluation helpers ----------

    def evaluate_and_save(self) -> Dict[str, Any]:
        if self.model is None or self.calibrator is None or self.val_thr_expected is None:
            raise RuntimeError("Model must be trained and calibrated before evaluation.")
        if self.test_df is None:
            raise RuntimeError("Test dataframe not loaded.")

        metrics_rows: List[Dict[str, Any]] = []
        thr = float(self.val_thr_expected)

        # Per-fold validation metrics
        for fold, (_, va_df) in sorted(self.folds.items()):
            X_va, y_va = prepare_xy_compat(va_df, self.target_col)
            y_val = y_va.astype(int)
            p_raw = self.model.predict_proba(X_va)[:, 1]
            p_raw = _clip01(p_raw)
            p_cal = _clip01(self.calibrator.transform(p_raw))
            
            pred = (p_cal >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
            auc = roc_auc_score(y_val, p_cal)
            pr_auc = average_precision_score(y_val, p_cal)
            logloss_ = log_loss(y_val, p_cal, labels=[0, 1])
            brier = brier_score_loss(y_val, p_cal)
            ks = ks_statistic(y_val.values, p_cal)
            prec_obs = tp / max(tp + fp, 1e-12)
            rec_obs = tp / max(tp + fn, 1e-12)
            acc_obs = (tp + tn) / max(len(y_val), 1e-12)
            f1_obs = (2 * prec_obs * rec_obs) / max(prec_obs + rec_obs, 1e-12)

            _, pos_true, neg_true = self._natural_counts("val", int(fold))
            w = class_weights_from_counts(y_val.values, pos_true, neg_true)
            exp = expected_threshold_metrics(y_val.values, p_cal, thr, pos_true, neg_true)
            pr_auc_w = float(average_precision_score(y_val, p_cal, sample_weight=w))
            logloss_w = weighted_logloss(y_val, p_cal, w)
            brier_w = weighted_brier(y_val, p_cal, w)

            p_base = pos_true / max(pos_true + neg_true, 1e-12)
            base_obs = prob_metrics(y_val.values, np.full_like(p_cal, p_base), None)
            base_obs["auc"] = 0.5
            base_exp = prob_metrics(y_val.values, np.full_like(p_cal, p_base), w)
            base_exp["auc"] = 0.5

            rate_pred_obs = float(pred.mean())
            base_cls_obs = random_baseline_expected_metrics(rate_pred_obs, pos_true=int((y_val == 1).sum()), neg_true=int((y_val == 0).sum()))
            base_cls_exp = random_baseline_expected_metrics(rate_pred_obs, pos_true=pos_true, neg_true=neg_true)

            row = dict(
                run_id=self.run_id,
                label=self.label_tag,
                split="val",
                fold=int(fold),
                n=len(y_val),
                pos=int(y_val.sum()),
                neg=int((y_val == 0).sum()),
                auc=float(auc),
                pr_auc=float(pr_auc),
                logloss=float(logloss_),
                brier=float(brier),
                ks=float(ks),
                precision=float(prec_obs),
                recall=float(rec_obs),
                accuracy=float(acc_obs),
                f1=float(f1_obs),
                threshold=thr,
                tp=int(tp),
                fp=int(fp),
                tn=int(tn),
                fn=int(fn),
                exp_pos=int(pos_true),
                exp_neg=int(neg_true),
                exp_pr_auc=float(pr_auc_w),
                exp_logloss=float(logloss_w),
                exp_brier=float(brier_w),
                exp_tp=float(exp["exp_tp"]),
                exp_fp=float(exp["exp_fp"]),
                exp_tn=float(exp["exp_tn"]),
                exp_fn=float(exp["exp_fn"]),
                exp_precision=float(exp["exp_precision"]),
                exp_recall=float(exp["exp_recall"]),
                exp_accuracy=float(exp["exp_accuracy"]),
                exp_f1=float(exp["exp_f1"]),
                tpr=float(exp["tpr"]),
                fpr=float(exp["fpr"]),
                base_prob_obs_auc=float(base_obs["auc"]),
                base_prob_obs_pr_auc=float(base_obs["pr_auc"]),
                base_prob_obs_logloss=float(base_obs["logloss"]),
                base_prob_obs_brier=float(base_obs["brier"]),
                base_prob_exp_auc=float(base_exp["auc"]),
                base_prob_exp_pr_auc=float(base_exp["pr_auc"]),
                base_prob_exp_logloss=float(base_exp["logloss"]),
                base_prob_exp_brier=float(base_exp["brier"]),
                base_rand_obs_tp=float(base_cls_obs["tp"]),
                base_rand_obs_fp=float(base_cls_obs["fp"]),
                base_rand_obs_tn=float(base_cls_obs["tn"]),
                base_rand_obs_fn=float(base_cls_obs["fn"]),
                base_rand_obs_precision=float(base_cls_obs["precision"]),
                base_rand_obs_recall=float(base_cls_obs["recall"]),
                base_rand_obs_accuracy=float(base_cls_obs["accuracy"]),
                base_rand_obs_f1=float(base_cls_obs["f1"]),
                base_rand_exp_tp=float(base_cls_exp["tp"]),
                base_rand_exp_fp=float(base_cls_exp["fp"]),
                base_rand_exp_tn=float(base_cls_exp["tn"]),
                base_rand_exp_fn=float(base_cls_exp["fn"]),
                base_rand_exp_precision=float(base_cls_exp["precision"]),
                base_rand_exp_recall=float(base_cls_exp["recall"]),
                base_rand_exp_accuracy=float(base_cls_exp["accuracy"]),
                base_rand_exp_f1=float(base_cls_exp["f1"]),
                base_rate_pred=float(rate_pred_obs),
                base_p_const=float(p_base),
            )
            metrics_rows.append(row)

        # Test metrics
        X_te, y_te = prepare_xy_compat(self.test_df, self.target_col)
        y_test = y_te.astype(int)
        p_raw_te = self.model.predict_proba(X_te)[:, 1]
        p_raw_te = _clip01(p_raw_te)
        p_te = self.calibrator.transform(p_raw_te)
        p_te = _clip01(p_te)
        
        pred_te = (p_te >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_te, labels=[0, 1]).ravel()
        auc_te = roc_auc_score(y_test, p_te)
        pr_auc_te = average_precision_score(y_test, p_te)
        logloss_te = log_loss(y_test, p_te, labels=[0, 1])
        brier_te = brier_score_loss(y_test, p_te)
        
        ks_te = ks_statistic(y_test.values, p_te)
        prec_te = tp / max(tp + fp, 1e-12)
        rec_te = tp / max(tp + fn, 1e-12)
        acc_te = (tp + tn) / max(len(y_test), 1e-12)
        f1_te = (2 * prec_te * rec_te) / max(prec_te + rec_te, 1e-12)

        _, pos_true_te, neg_true_te = self._natural_counts("test", 0)
        w_te = class_weights_from_counts(y_test.values, pos_true_te, neg_true_te)
        exp_te = expected_threshold_metrics(y_test.values, p_te, thr, pos_true_te, neg_true_te)
        pr_auc_te_w = float(average_precision_score(y_test, p_te, sample_weight=w_te))
        logloss_te_w = weighted_logloss(y_test, p_te, w_te)
        brier_te_w = weighted_brier(y_test, p_te, w_te)

        p_base_te = pos_true_te / max(pos_true_te + neg_true_te, 1e-12)
        base_obs_te = prob_metrics(y_test.values, np.full_like(p_te, p_base_te), None)
        base_obs_te["auc"] = 0.5
        base_exp_te = prob_metrics(y_test.values, np.full_like(p_te, p_base_te), w_te)
        base_exp_te["auc"] = 0.5

        rate_pred_te = float(pred_te.mean())
        base_cls_obs_te = random_baseline_expected_metrics(rate_pred_te, pos_true=int((y_test == 1).sum()), neg_true=int((y_test == 0).sum()))
        base_cls_exp_te = random_baseline_expected_metrics(rate_pred_te, pos_true=pos_true_te, neg_true=neg_true_te)

        metrics_rows.append(
            dict(
                run_id=self.run_id,
                label=self.label_tag,
                split="test",
                fold=0,
                n=len(y_test),
                pos=int(y_test.sum()),
                neg=int((y_test == 0).sum()),
                auc=float(auc_te),
                pr_auc=float(pr_auc_te),
                logloss=float(logloss_te),
                brier=float(brier_te),
                ks=float(ks_te),
                precision=float(prec_te),
                recall=float(rec_te),
                accuracy=float(acc_te),
                f1=float(f1_te),
                threshold=thr,
                tp=int(tp),
                fp=int(fp),
                tn=int(tn),
                fn=int(fn),
                exp_pos=int(pos_true_te),
                exp_neg=int(neg_true_te),
                exp_pr_auc=float(pr_auc_te_w),
                exp_logloss=float(logloss_te_w),
                exp_brier=float(brier_te_w),
                exp_tp=float(exp_te["exp_tp"]),
                exp_fp=float(exp_te["exp_fp"]),
                exp_tn=float(exp_te["exp_tn"]),
                exp_fn=float(exp_te["exp_fn"]),
                exp_precision=float(exp_te["exp_precision"]),
                exp_recall=float(exp_te["exp_recall"]),
                exp_accuracy=float(exp_te["exp_accuracy"]),
                exp_f1=float(exp_te["exp_f1"]),
                tpr=float(exp_te["tpr"]),
                fpr=float(exp_te["fpr"]),
                base_prob_obs_auc=float(base_obs_te["auc"]),
                base_prob_obs_pr_auc=float(base_obs_te["pr_auc"]),
                base_prob_obs_logloss=float(base_obs_te["logloss"]),
                base_prob_obs_brier=float(base_obs_te["brier"]),
                base_prob_exp_auc=float(base_exp_te["auc"]),
                base_prob_exp_pr_auc=float(base_exp_te["pr_auc"]),
                base_prob_exp_logloss=float(base_exp_te["logloss"]),
                base_prob_exp_brier=float(base_exp_te["brier"]),
                base_rand_obs_tp=float(base_cls_obs_te["tp"]),
                base_rand_obs_fp=float(base_cls_obs_te["fp"]),
                base_rand_obs_tn=float(base_cls_obs_te["tn"]),
                base_rand_obs_fn=float(base_cls_obs_te["fn"]),
                base_rand_obs_precision=float(base_cls_obs_te["precision"]),
                base_rand_obs_recall=float(base_cls_obs_te["recall"]),
                base_rand_obs_accuracy=float(base_cls_obs_te["accuracy"]),
                base_rand_obs_f1=float(base_cls_obs_te["f1"]),
                base_rand_exp_tp=float(base_cls_exp_te["tp"]),
                base_rand_exp_fp=float(base_cls_exp_te["fp"]),
                base_rand_exp_tn=float(base_cls_exp_te["tn"]),
                base_rand_exp_fn=float(base_cls_exp_te["fn"]),
                base_rand_exp_precision=float(base_cls_exp_te["precision"]),
                base_rand_exp_recall=float(base_cls_exp_te["recall"]),
                base_rand_exp_accuracy=float(base_cls_exp_te["accuracy"]),
                base_rand_exp_f1=float(base_cls_exp_te["f1"]),
                base_rate_pred=float(rate_pred_te),
                base_p_const=float(p_base_te),
            )
        )

        metrics_df = pd.DataFrame(metrics_rows).sort_values(["split", "fold"])
        csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
        gcs_upload_bytes(f"{self.gcs_prefix}/metrics_per_fold_and_test.csv", csv_bytes, content_type="text/csv")
        self.metrics_df = metrics_df

        # Plots and supplementary tables (test set)
        plot_roc(y_test.values, p_te, thr, f"{self.gcs_prefix}/roc_curve_test.png")
        plot_pr(y_test.values, p_te, thr, f"{self.gcs_prefix}/pr_curve_test.png")

        calib = self._calibration_bins(y_test.values, p_te, bins=20)
        gcs_upload_bytes(
            f"{self.gcs_prefix}/calibration_bins_test_observed.csv",
            calib.to_csv(index=False).encode("utf-8"),
            content_type="text/csv",
        )
        plot_calibration(
            calib, "mean_pred", "rate",
            f"{self.gcs_prefix}/calibration_test_observed.png",
            "Calibration (observed, test)"
        )


        calib_w = self._calibration_bins_weighted(y_test.values, p_te, w_te, bins=20)
        gcs_upload_bytes(f"{self.gcs_prefix}/calibration_bins_test_expected.csv",
                         calib_w.to_csv(index=False).encode("utf-8"), content_type="text/csv")
        plot_calibration(calib_w, "mean_pred_w", "rate_w", f"{self.gcs_prefix}/calibration_test_expected.png", "Calibration (expected, test)")

        lift = self._lift_table(y_test.values, p_te, deciles=10)
        gcs_upload_bytes(f"{self.gcs_prefix}/lift_table_test_observed.csv",
                         lift.to_csv(index=False).encode("utf-8"), content_type="text/csv")
        plot_lift(lift, "decile", "lift", f"{self.gcs_prefix}/lift_test_observed.png", "Lift by decile (observed, test)")
        
        lift_w = self._lift_table_weighted(y_test.values, p_te, w_te, deciles=10)
        gcs_upload_bytes(f"{self.gcs_prefix}/lift_table_test_expected.csv",
                         lift_w.to_csv(index=False).encode("utf-8"), content_type="text/csv")
        plot_lift(lift_w, "decile", "lift_w", f"{self.gcs_prefix}/lift_test_expected.png", "Lift by decile (expected, test)")

        cm_obs_model = np.array([[tn, fp], [fn, tp]], dtype=float)
        base_obs_cm = np.array([
            [base_cls_obs_te["tn"], base_cls_obs_te["fp"]],
            [base_cls_obs_te["fn"], base_cls_obs_te["tp"]],
        ], dtype=float)
        plot_confusions_side_by_side(cm_obs_model, base_obs_cm,
            ("Model (observed)", "Random baseline (observed)"),
            f"{self.gcs_prefix}/confusion_test_observed_side_by_side.png",
            "Confusion matrices (observed, test)")

        cm_exp_model = np.array([
            [exp_te["exp_tn"], exp_te["exp_fp"]],
            [exp_te["exp_fn"], exp_te["exp_tp"]],
        ], dtype=float)
        cm_exp_base = np.array([
            [base_cls_exp_te["tn"], base_cls_exp_te["fp"]],
            [base_cls_exp_te["fn"], base_cls_exp_te["tp"]],
        ], dtype=float)
        plot_confusions_side_by_side(cm_exp_model, cm_exp_base,
            ("Model (expected)", "Random baseline (expected)"),
            f"{self.gcs_prefix}/confusion_test_expected_side_by_side.png",
            "Confusion matrices (expected, test)")

        model_obs_metrics = {"precision": prec_te, "recall": rec_te, "accuracy": acc_te, "f1": f1_te}
        base_obs_metrics = {k: base_cls_obs_te[k] for k in ["precision", "recall", "accuracy", "f1"]}
        plot_metric_bars(model_obs_metrics, base_obs_metrics, "Classification metrics (observed, test)",
            f"{self.gcs_prefix}/classification_bars_test_observed.png")

        model_exp_metrics = {
            "precision": exp_te["exp_precision"],
            "recall": exp_te["exp_recall"],
            "accuracy": exp_te["exp_accuracy"],
            "f1": exp_te["exp_f1"],
        }
        base_exp_metrics = {
            "precision": base_cls_exp_te["precision"],
            "recall": base_cls_exp_te["recall"],
            "accuracy": base_cls_exp_te["accuracy"],
            "f1": base_cls_exp_te["f1"],
        }
        plot_metric_bars(model_exp_metrics, base_exp_metrics, "Classification metrics (expected, test)",
            f"{self.gcs_prefix}/classification_bars_test_expected.png")

        plot_score_hist(y_test.values, p_te, f"{self.gcs_prefix}/score_hist_test.png")
        try:
            plot_feature_importance(self.model.get_booster(), f"{self.gcs_prefix}/feature_importance_top20.png", topn=20)
        except Exception:
            pass

        summary = {
            "label": self.label_tag,
            "best_params": self.best_params,
            "threshold_chosen_on_validation_expectedF1": thr,
            "observed_test": {
                "auc": float(auc_te),
                "pr_auc": float(pr_auc_te),
                "logloss": float(logloss_te),
                "brier": float(brier_te),
                "ks": float(ks_te),
                "precision": float(prec_te),
                "recall": float(rec_te),
                "accuracy": float(acc_te),
                "f1": float(f1_te),
                "n": int(len(y_test)),
                "pos": int(y_test.sum()),
                "neg": int((y_test == 0).sum()),
                "baseline_const_prob": {
                    "auc": 0.5,
                    "pr_auc": float(base_obs_te["pr_auc"]),
                    "logloss": float(base_obs_te["logloss"]),
                    "brier": float(base_obs_te["brier"]),
                },
                "baseline_random_same_rate": {
                    "rate_pred": rate_pred_te,
                    "precision": float(base_cls_obs_te["precision"]),
                    "recall": float(base_cls_obs_te["recall"]),
                    "f1": float(base_cls_obs_te["f1"]),
                    "accuracy": float(base_cls_obs_te["accuracy"]),
                },
            },
            "expected_test": {
                "pr_auc": float(pr_auc_te_w),
                "logloss": float(logloss_te_w),
                "brier": float(brier_te_w),
                "f1": float(exp_te["exp_f1"]),
                "precision": float(exp_te["exp_precision"]),
                "recall": float(exp_te["exp_recall"]),
                "accuracy": float(exp_te["exp_accuracy"]),
                "tp": float(exp_te["exp_tp"]),
                "fp": float(exp_te["exp_fp"]),
                "tn": float(exp_te["exp_tn"]),
                "fn": float(exp_te["exp_fn"]),
                "baseline_const_prob": {
                    "auc": 0.5,
                    "pr_auc": float(base_exp_te["pr_auc"]),
                    "logloss": float(base_exp_te["logloss"]),
                    "brier": float(base_exp_te["brier"]),
                },
                "baseline_random_same_rate": {
                    "rate_pred": rate_pred_te,
                    "precision": float(base_cls_exp_te["precision"]),
                    "recall": float(base_cls_exp_te["recall"]),
                    "f1": float(base_cls_exp_te["f1"]),
                    "accuracy": float(base_cls_exp_te["accuracy"]),
                },
            },
            "run_id": self.run_id,
        }

        gcs_upload_json(f"{self.gcs_prefix}/summary.json", summary, indent=2)
        self.logger.info(
            "Saved evaluation artifacts for label=%s at %s",
            self.label_tag,
            self.gcs_prefix,
        )

        return {
            "metrics_df": metrics_df,
            "summary": summary,
            # We no longer return local paths; all artifacts live under self.gcs_prefix
        }

    def _natural_counts(self, split: str, fold: int) -> Tuple[int, int, int]:
        """Return (n, pos, neg) for the specified split/fold from metadata."""
        if self.meta_counts is None:
            raise RuntimeError("meta_counts not loaded.")
        subset = self.meta_counts[(self.meta_counts["split"] == split) & (self.meta_counts["fold"] == fold)]
        if subset.empty:
            raise RuntimeError(f"No natural counts for label={self.label_tag} split={split} fold={fold}")
        n = int(subset["n"].iloc[0])
        pos = int(subset["pos"].iloc[0])
        neg = int(subset["neg"].iloc[0])
        return n, pos, neg

    def _natural_totals(self, split: str) -> Tuple[int, int]:
        """Return total expected positives/negatives across folds for a split."""
        if self.meta_counts is None:
            raise RuntimeError("meta_counts not loaded.")
        m2 = self.meta_counts[self.meta_counts["split"] == split]
        pos_true = int(m2["pos"].sum())
        neg_true = int(m2["neg"].sum())
        return pos_true, neg_true

    @staticmethod
    def _scale_pos_weight(y: pd.Series) -> float:
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        return (neg / max(pos, 1.0))

    def _calibration_bins(self, y_true: np.ndarray, y_prob: np.ndarray, bins: int = 20) -> pd.DataFrame:
        df = pd.DataFrame({"y": y_true, "p": y_prob})
        df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
        return df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), rate=("y", "mean")).reset_index()

    def _calibration_bins_weighted(self, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, bins: int = 20) -> pd.DataFrame:
        df = pd.DataFrame({"y": y_true, "p": y_prob, "w": w})
        df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
        g = df.groupby("bin")
        out = g.apply(lambda d: pd.Series({
            "n_w": d["w"].sum(),
            "mean_pred_w": np.average(d["p"], weights=d["w"]),
            "rate_w": np.average(d["y"], weights=d["w"]),
        })).reset_index()
        return out

    def _lift_table(self, y_true: np.ndarray, y_prob: np.ndarray, deciles: int = 10) -> pd.DataFrame:
        df = pd.DataFrame({"y": y_true, "p": y_prob})
        df["decile"] = pd.qcut(df["p"], q=deciles, labels=False, duplicates="drop")
        tbl = df.groupby("decile").agg(
            n=("y", "size"),
            responders=("y", "sum"),
            avg_score=("p", "mean"),
            rate=("y", "mean"),
        ).reset_index()
        overall_rate = df["y"].mean() if len(df) else 0.0
        tbl["lift"] = tbl["rate"] / max(overall_rate, 1e-12)
        return tbl.sort_values("decile", ascending=False).reset_index(drop=True)

    def _lift_table_weighted(self, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, deciles: int = 10) -> pd.DataFrame:
        df = pd.DataFrame({"y": y_true, "p": y_prob, "w": w})
        df["decile"] = pd.qcut(df["p"], q=deciles, labels=False, duplicates="drop")
        g = df.groupby("decile")
        tbl = g.apply(lambda d: pd.Series({
            "n_w": d["w"].sum(),
            "responders_w": np.sum(d["w"] * d["y"]),
            "avg_score_w": np.average(d["p"], weights=d["w"]),
            "rate_w": np.average(d["y"], weights=d["w"]),
        })).reset_index()
        overall_rate_w = np.average(y_true, weights=w) if len(df) else 0.0
        tbl["lift_w"] = tbl["rate_w"] / max(overall_rate_w, 1e-12)
        return tbl.sort_values("decile", ascending=False).reset_index(drop=True)

    # ---------- summarize ----------

    def summary(self) -> TrainOutputs:
        # quick mean val AUC using fitted model and current folds (for reporting)
        aucs = []
        for _, (_, va_df) in self.folds.items():
            X_va, y_va = prepare_xy_compat(va_df, self.target_col)
            p = self.model.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, p))
        auc_mean = float(np.mean(aucs))
        return TrainOutputs(
            label=self.label_tag,
            best_params=self.best_params or {},
            threshold_expected=self.val_thr_expected or 0.5,
            auc_val_mean=auc_mean,
            artifact_gcs_prefix=str(self.gcs_prefix),
        )


# -----------------------
# Public API
# -----------------------

def run_training(
    *,
    project_id: str,
    region: str,
    dataset: str,
    train_table: str,
    metrics_table: str,
    labels_yaml: str,
    thresholds_policy: str,
    gcs_model_bucket: str,
    vertex_model_display_name: str,
    vertex_model_registry_label: str,
    artifact_repo: Optional[str],
    run_id: str,
    primary_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrate per-label training consistent with the original local script.
    Returns a summary dict with per-label outputs. Artifact uploads and metric
    persistence occur in later steps (artifact_io, metrics_to_bq).
    """
    global PROJECT_ID, BQ_LOCATION, BQ_DATASET, TABLE_DATA, TABLE_META, TABLE_PREP
    PROJECT_ID = project_id
    BQ_LOCATION = region
    BQ_DATASET = dataset
    TABLE_DATA = f"{PROJECT_ID}.{BQ_DATASET}.{train_table}"
    TABLE_META = f"{PROJECT_ID}.{BQ_DATASET}.cv_build_metadata"
    TABLE_PREP = f"{PROJECT_ID}.{BQ_DATASET}.prep_metadata"


    logger = get_logger("pts.training.train")

    # Load labels config
    with open(labels_yaml, "r", encoding="utf-8") as f:
        import yaml
        labels_cfg = yaml.safe_load(f) or {}
    labels = [l["id"] for l in labels_cfg.get("labels", []) if l.get("enabled", True)]
    if primary_label and primary_label not in labels:
        labels.insert(0, primary_label)


    results: Dict[str, Any] = {"run_id": run_id, "labels": {}}

    for tag in labels:
        logger.info("=== Training label: %s ===", tag)
        trainer = LabelTrainer(
            label_tag=tag,
            run_id=run_id,
            gcs_model_bucket=gcs_model_bucket,
            vertex_model_display_name=vertex_model_display_name,
            vertex_model_registry_label=vertex_model_registry_label,
        )
        trainer.load()
    
        logger.info("Bayesian optimization starting...")
        trainer.bayes_optimize()
    
        logger.info("Fitting final model + calibration + threshold selection...")
        trainer.fit_final()
    
        eval_outputs = trainer.evaluate_and_save()
        summary = trainer.summary()
    
        results["labels"][tag] = {
            "best_params": summary.best_params,
            "threshold_expected": summary.threshold_expected,
            "auc_val_mean": summary.auc_val_mean,
            "artifact_gcs_prefix": summary.artifact_gcs_prefix,
        }
    
        # Register version & write manifest pointing at GCS artifacts
        if artifact_io is not None:
            try:
                artifact_io.write_manifest_and_register_existing(
                    label_tag=tag,
                    run_id=run_id,
                    gcs_model_bucket=gcs_model_bucket,
                    vertex_model_display_name=vertex_model_display_name,
                    vertex_model_registry_label=vertex_model_registry_label,
                    best_params=summary.best_params,
                    threshold=summary.threshold_expected,
                )
            except Exception as e:
                logger.warning("artifact_io.write_manifest_and_register_existing failed: %s", e)
    
        # Persist metrics to BigQuery
        if metrics_to_bq is not None:
            try:
                metrics_to_bq.write_training_summary(
                    project_id=project_id,
                    dataset=dataset,
                    table=metrics_table,
                    run_id=run_id,
                    label_tag=tag,
                    auc_val_mean=summary.auc_val_mean,
                    threshold_expected=summary.threshold_expected,
                    params=summary.best_params,
                )
                metrics_df = eval_outputs.get("metrics_df")
                if metrics_df is not None:
                    metrics_to_bq.write_metrics_detail(
                        project_id=project_id,
                        dataset=dataset,
                        table=metrics_table,
                        metrics_df=metrics_df,
                    )
            except Exception as e:
                logger.warning("metrics_to_bq write failed: %s", e)



    logger.info("Training summary: %s", json.dumps(results, indent=2))
    return results
