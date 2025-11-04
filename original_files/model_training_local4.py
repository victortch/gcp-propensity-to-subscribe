#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Economedia - Propensity to Subscribe
# - Uses latest RUN_ID from cv_build_metadata
# - Trains one model per label (cap_90d, dne_90d)
# - Drops id/meta fields from features
# - Reports observed (balanced) and expected (true prevalence) metrics
# - Baselines: constant-prob (probability metrics) + random same-rate (classification metrics)
# - NEW:
#   (a) Threshold chosen to MAXIMIZE EXPECTED F1 on VALIDATION using metadata counts
#   (b) Weighted ISOTONIC calibration (validation-based) applied to probabilities
#   (c) Confusion matrices saved side-by-side with RANDOM BASELINE (observed & expected)
#   (d) Bar charts comparing classification metrics (precision/recall/accuracy/F1) against baseline

import os
os.chdir('/Users/victortchervenobrejki/Documents/Economedia/propensity-to-subscribe')

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from joblib import dump
from bayes_opt import BayesianOptimization
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, f1_score,
    confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.isotonic import IsotonicRegression

from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig
from google.oauth2 import service_account


# =========================
# Config
# =========================
SERVICE_ACCOUNT_KEY = "/Users/victortchervenobrejki/Documents/Economedia/economedia-data-prod-laoy-2cd58f972a4e.json"
PROJECT_ID = "economedia-data-prod-laoy"
BQ_LOCATION = "europe-west3"
DATASET = "propensity_to_subscribe"
TABLE_DATA = f"{PROJECT_ID}.{DATASET}.train_data"          # rows produced by data-engineering step
TABLE_META = f"{PROJECT_ID}.{DATASET}.cv_build_metadata"   # natural counts & run provenance

# Train both media
LABELS: List[str] = [#"cap_90d", 
                     #"dne_90d",
                     "cap_30d", 
                     "dne_30d"]

RANDOM_STATE = 42
N_BAYES_INIT = 5
N_BAYES_ITER = 55

RUN_DIR = Path(f"./runs_local/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
RUN_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# BQ Client
# =========================
def make_bq_client():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_KEY,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(project=PROJECT_ID, credentials=creds, location=BQ_LOCATION)


# =========================
# Load latest run + data + metadata
# =========================
def latest_run_id(client: bigquery.Client) -> str:
    sql = f"""
    SELECT run_id
    FROM `{TABLE_META}`
    ORDER BY run_started_at DESC
    LIMIT 1
    """
    rid = client.query(sql).result().to_dataframe().iloc[0, 0]
    if not isinstance(rid, str) or not rid:
        raise RuntimeError("Could not find latest run_id in metadata table.")
    return rid


def load_train_table(client: bigquery.Client, run_id: str, label_tag: str) -> pd.DataFrame:
    sql = f"""
    SELECT *
    FROM `{TABLE_DATA}`
    WHERE run_id = @rid AND label = @label
    """
    job_config = QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("rid", "STRING", run_id),
        bigquery.ScalarQueryParameter("label", "STRING", label_tag),
    ])
    df = client.query(sql, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_meta_counts(client: bigquery.Client, run_id: str, label_tag: str) -> pd.DataFrame:
    """
    Returns rows for natural counts for val folds and test.
    Columns: split, fold, n, pos, neg
    """
    sql = f"""
    SELECT split, CAST(fold AS INT64) AS fold, n, pos, neg
    FROM `{TABLE_META}`
    WHERE run_id = @rid AND label_tag = @label AND split IN ('val','test')
    """
    job_config = QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("rid", "STRING", run_id),
        bigquery.ScalarQueryParameter("label", "STRING", label_tag),
    ])
    df = client.query(sql, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)
    if df.empty:
        raise RuntimeError(f"No metadata counts found for run_id={run_id}, label={label_tag}")
    return df


# =========================
# Feature Prep
# =========================
NON_FEATURE_COLS_BASE = {
    "user_id", "date", "split", "fold", "label", "run_id",  # drop these always
}

def prepare_xy(df: pd.DataFrame, target_col: str = "y") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build X, y. Drops id/meta columns and anything ending with '_id'.
    Cast numerics to float32, categories to codes; fillna=0.
    """
    dynamic_id_cols = {c for c in df.columns if c.lower().endswith("_id")}
    drop_cols = NON_FEATURE_COLS_BASE | dynamic_id_cols
    drop_cols |= {c for c in df.columns if c.startswith("y_")}  # drop all y_* from features

    feat_cols = [c for c in df.columns if c not in drop_cols and c != target_col]

    X = df[feat_cols].copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
        else:
            X[c] = X[c].astype("category").cat.codes.astype("float32")
    X = X.fillna(0.0)

    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} not in dataframe.")
    y = df[target_col].astype("Int64").astype("float32")
    return X, y


def scale_pos_weight(y: pd.Series) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return (neg / max(pos, 1.0))


# =========================
# Metrics & Adjustments
# =========================
def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    pos = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    neg = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return float((pos - neg).abs().max())


def weighted_logloss(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray) -> float:
    return float(log_loss(y_true, y_prob, sample_weight=w, labels=[0, 1]))


def weighted_brier(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray) -> float:
    num = np.average((y_true - y_prob) ** 2, weights=w)
    return float(num)


def class_weights_from_counts(y_true: np.ndarray, pos_true: int, neg_true: int) -> np.ndarray:
    """Weights so that sum(weights over pos)=pos_true and sum(weights over neg)=neg_true."""
    pos_s = max(1, int((y_true == 1).sum()))
    neg_s = max(1, int((y_true == 0).sum()))
    w_pos = pos_true / pos_s
    w_neg = neg_true / neg_s
    return np.where(y_true == 1, w_pos, w_neg).astype("float64")


def expected_counts_from_rates(tpr: float, fpr: float, pos_true: int, neg_true: int) -> Tuple[float, float, float, float]:
    tp = tpr * pos_true
    fn = (1.0 - tpr) * pos_true
    fp = fpr * neg_true
    tn = (1.0 - fpr) * neg_true
    return tp, fp, tn, fn


def expected_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float,
                               pos_true: int, neg_true: int) -> Dict[str, float]:
    """Expected confusion & derived metrics at 'thr' for true prevalence."""
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
        "exp_tp": float(tp), "exp_fp": float(fp), "exp_tn": float(tn), "exp_fn": float(fn),
        "exp_precision": float(prec), "exp_recall": float(rec),
        "exp_accuracy": float(acc), "exp_f1": float(f1),
        "tpr": float(tpr), "fpr": float(fpr),
    }


def prob_metrics(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray | None = None) -> Dict[str, float]:
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
    """Random classifier that predicts positives at rate_pred (same as our model)."""
    tp = rate_pred * pos_true
    fp = rate_pred * neg_true
    fn = (1 - rate_pred) * pos_true
    tn = (1 - rate_pred) * neg_true
    n = pos_true + neg_true
    prec = tp / max(tp + fp, 1e-12)
    rec = tp / max(tp + fn, 1e-12)
    acc = (tp + tn) / max(n, 1e-12)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall": rec, "accuracy": acc, "f1": f1}


def best_expected_f1_threshold_global(y_true: np.ndarray,
                                      y_prob: np.ndarray,
                                      pos_true: int,
                                      neg_true: int,
                                      grid: int = 200) -> Tuple[float, float]:
    """
    Scan thresholds; compute TPR/FPR on provided sample; convert to expected counts
    with (pos_true, neg_true); return threshold maximizing expected F1.
    """
    ts = np.linspace(0.0, 1.0, grid)
    best_t, best_val = 0.5, -1.0
    for t in ts:
        pred = (y_prob >= t).astype(int)
        tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        pos_s = max(1, int((y_true == 1).sum()))
        neg_s = max(1, int((y_true == 0).sum()))
        tpr = tp_s / pos_s
        fpr = fp_s / neg_s
        tp = tpr * pos_true
        fp = fpr * neg_true
        fn = (1 - tpr) * pos_true
        # tn = (1 - fpr) * neg_true
        prec = tp / max(tp + fp, 1e-12)
        rec  = tp / max(tp + fn, 1e-12)
        f1   = (2 * prec * rec) / max(prec + rec, 1e-12)
        if f1 > best_val:
            best_val, best_t = f1, t
    return float(best_t), float(best_val)


# =========================
# Visualization helpers
# =========================
def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, thr: float, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    tpr_point = tp / max(tp + fn, 1e-12)
    fpr_point = fp / max(fp + tn, 1e-12)

    plt.figure()
    plt.plot(fpr, tpr, label="Model ROC")
    plt.plot([0,1], [0,1], linestyle="--", label="Random")
    plt.scatter([fpr_point], [tpr_point], marker="o", label=f"Thr={thr:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, thr: float, out_path: Path):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    prevalence = float(np.mean(y_true))
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
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
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_calibration(calib_df: pd.DataFrame, mean_pred_col: str, rate_col: str, out_path: Path, title: str):
    plt.figure()
    x = calib_df[mean_pred_col].values
    y = calib_df[rate_col].values
    plt.plot([0,1], [0,1], linestyle="--", label="Perfect")
    plt.scatter(x, y, label="Bins")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_lift(tbl: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str):
    plt.figure()
    plt.plot(tbl[x_col], tbl[y_col], marker="o", label="Model")
    plt.hlines(1.0, tbl[x_col].min(), tbl[x_col].max(), linestyles="--", label="Baseline")
    plt.xlabel("Decile (10=top)")
    plt.ylabel("Lift")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _annotate_cm(ax, cm: np.ndarray):
    ax.imshow(cm, interpolation="nearest")
    for (i, j), v in np.ndenumerate(cm):
        if float(v).is_integer():
            txt = f"{int(v)}"
        else:
            txt = f"{v:.1f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0,1])
    ax.set_yticklabels(["True 0", "True 1"])


def plot_confusions_side_by_side(cm_left: np.ndarray, cm_right: np.ndarray,
                                 titles: Tuple[str, str], out_path: Path, suptitle: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    _annotate_cm(axes[0], cm_left)
    axes[0].set_title(titles[0])
    _annotate_cm(axes[1], cm_right)
    axes[1].set_title(titles[1])
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_score_hist(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, bins: int = 40):
    plt.figure()
    plt.hist(y_prob[y_true==0], bins=bins, alpha=0.6, label="Negatives")
    plt.hist(y_prob[y_true==1], bins=bins, alpha=0.6, label="Positives")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Score distribution by class (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(booster, out_path: Path, topn: int = 20):
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
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_bars(model_metrics: Dict[str, float],
                     baseline_metrics: Dict[str, float],
                     title: str, out_path: Path):
    metrics = ["precision", "recall", "accuracy", "f1"]
    model_vals = [float(model_metrics.get(m, np.nan)) for m in metrics]
    base_vals  = [float(baseline_metrics.get(m, np.nan)) for m in metrics]

    x = np.arange(len(metrics))
    w = 0.38
    plt.figure()

    bars_model = plt.bar(x - w/2, model_vals, width=w, label="Model")
    bars_base  = plt.bar(x + w/2, base_vals,  width=w, label="Random baseline")

    # add labels on bars
    def _autolabel(bars):
        for b in bars:
            h = b.get_height()
            if np.isnan(h):
                label = "NA"
                h_plot = 0.0
            else:
                label = f"{h:.2f}"
                h_plot = h
            plt.text(
                b.get_x() + b.get_width()/2,
                h_plot + 0.01,                # a little above the bar
                label,
                ha="center",
                va="bottom",
                fontsize=9
            )

    _autolabel(bars_model)
    _autolabel(bars_base)

    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.ylim(0, 1.05)  # small headroom for labels
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



# =========================
# Trainer
# =========================
class LabelTrainer:
    def __init__(self, label_tag: str, out_dir: Path):
        self.label_tag = label_tag
        self.target_col = "y"
        self.out_dir = out_dir / label_tag
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.client = make_bq_client()
        self.run_id = latest_run_id(self.client)

        self.folds = {}           # fold -> (train_df, val_df)
        self.test_df = None
        self.meta_counts = None   # DataFrame with natural counts per fold

        self.best_params = None
        self.model = None
        self.calibrator: IsotonicRegression | None = None
        self.val_thr_expected: float | None = None  # global threshold chosen on validation (expected F1)

    def load(self):
        df = load_train_table(self.client, self.run_id, self.label_tag)
        # Partition by split/fold
        self.test_df = df[df["split"] == "test"].copy()
        self.folds = {}
        for k in sorted(df.loc[df["split"].isin(["train","val"]), "fold"].dropna().unique()):
            tr = df[(df["split"] == "train") & (df["fold"] == k)].copy()
            va = df[(df["split"] == "val")   & (df["fold"] == k)].copy()
            if not tr.empty and not va.empty:
                self.folds[int(k)] = (tr, va)
        if not self.folds:
            raise RuntimeError(f"No train/val folds found for {self.label_tag}")

        # Natural counts from metadata
        self.meta_counts = load_meta_counts(self.client, self.run_id, self.label_tag)
        print(f"[info] {self.label_tag}: run_id={self.run_id}, folds={len(self.folds)}, test_rows={len(self.test_df):,}")

    def _cv_score(self, max_depth, gamma, colsample_bytree, learning_rate, n_estimators, reg_alpha, reg_lambda):
        """Objective: mean ROC AUC across folds."""
        params = dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            enable_categorical=False,
            max_depth=int(round(max_depth)),
            gamma=float(gamma),
            colsample_bytree=float(colsample_bytree),
            learning_rate=float(learning_rate),
            n_estimators=int(round(n_estimators)),
            reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda),
        )
        aucs = []
        for _, (tr_df, va_df) in self.folds.items():
            X_tr, y_tr = prepare_xy(tr_df, self.target_col)
            X_va, y_va = prepare_xy(va_df, self.target_col)
            spw = scale_pos_weight(y_tr)
            clf = xgb.XGBClassifier(**params, scale_pos_weight=spw)
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, p))
        return float(np.mean(aucs)) if aucs else 0.0

    def bayes_optimize(self):
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
        self.best_params = optimizer.max["params"]
        self.best_params["max_depth"] = int(round(self.best_params["max_depth"]))
        self.best_params["n_estimators"] = int(round(self.best_params["n_estimators"]))
        with open(self.out_dir / "best_params.json", "w") as f:
            json.dump(self.best_params, f, indent=2)
        print(f"[info] {self.label_tag} best params: {self.best_params}")

    def fit_final(self):
        # train on ALL train folds combined
        X_list, y_list = [], []
        for k, (tr_df, _) in self.folds.items():
            X_tr, y_tr = prepare_xy(tr_df, self.target_col)
            X_list.append(X_tr); y_list.append(y_tr)
        X_all = pd.concat(X_list, axis=0)
        y_all = pd.concat(y_list, axis=0)
        spw = scale_pos_weight(y_all)
        params = dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            enable_categorical=False,
            scale_pos_weight=spw,
            **self.best_params,
        )
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_all, y_all)
        dump(self.model, self.out_dir / f"xgb_{self.label_tag}.joblib")
        print(f"[info] {self.label_tag}: model saved -> {self.out_dir / f'xgb_{self.label_tag}.joblib'}")

    # ---------- NEW: Calibration + expected-threshold selection on validation ----------
    def calibrate_and_choose_expected_threshold(self):
        """Fit isotonic calibrator on validation (weighted) and choose global threshold maximizing expected F1."""
        # 1) Collect validation predictions from the final model
        p_val_all, y_val_all = [], []
        pos_true_total, neg_true_total = 0, 0

        for k, (_, va_df) in self.folds.items():
            X_va, y_va = prepare_xy(va_df, self.target_col)
            p_raw = self.model.predict_proba(X_va)[:, 1]
            p_val_all.append(p_raw); y_val_all.append(y_va.values)

            # sum true counts from metadata for expected weighting
            _, pos_true_k, neg_true_k = self._natural_counts("val", int(k))
            pos_true_total += int(pos_true_k)
            neg_true_total += int(neg_true_k)

        p_val_all = np.concatenate(p_val_all)
        y_val_all = np.concatenate(y_val_all)

        # 2) Weighted isotonic calibration
        w_val = class_weights_from_counts(y_val_all, pos_true_total, neg_true_total)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_val_all, y_val_all, sample_weight=w_val)
        dump(iso, self.out_dir / f"isotonic_calibrator_{self.label_tag}.joblib")
        self.calibrator = iso

        p_val_cal = iso.transform(p_val_all)

        # 3) Choose threshold that maximizes EXPECTED F1 on validation (global)
        thr, best_exp_f1 = best_expected_f1_threshold_global(
            y_true=y_val_all,
            y_prob=p_val_cal,
            pos_true=pos_true_total,
            neg_true=neg_true_total,
            grid=400
        )
        self.val_thr_expected = float(thr)
        print(f"[info] {self.label_tag}: chosen global validation threshold (expected F1) = {self.val_thr_expected:.4f}; "
              f"expected F1 on validation={best_exp_f1:.4f}")

    # ----------------------------------------------------------------------

    def _natural_counts(self, split: str, fold: int) -> Tuple[int, int, int]:
        """Returns (n_true, pos_true, neg_true) for given split/fold from metadata."""
        m = self.meta_counts
        row = m[(m["split"] == split) & (m["fold"] == fold)]
        if row.empty:
            raise RuntimeError(f"No natural counts for {self.label_tag} split={split} fold={fold}")
        n_true = int(row["n"].iloc[0])
        pos_true = int(row["pos"].iloc[0])
        neg_true = int(row["neg"].iloc[0])
        return n_true, pos_true, neg_true

    def _baseline_constant_prob(self, y_true: np.ndarray, p_const: float, weights: np.ndarray | None):
        y_prob = np.full_like(y_true, fill_value=p_const, dtype=float)
        return prob_metrics(y_true, y_prob, w=weights)

    def evaluate_and_save(self):
        assert self.calibrator is not None and self.val_thr_expected is not None, \
            "Call calibrate_and_choose_expected_threshold() before evaluate_and_save()."

        # ---------- Per-fold validation metrics (using calibrated probs + global val threshold) ----------
        metrics_rows = []
        for k, (_, va_df) in self.folds.items():
            X_va, y_va = prepare_xy(va_df, self.target_col)
            p_raw = self.model.predict_proba(X_va)[:, 1]
            p = self.calibrator.transform(p_raw)
            thr = self.val_thr_expected

            # Observed (balanced val)
            pred = (p >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_va, pred, labels=[0,1]).ravel()
            auc = roc_auc_score(y_va, p)
            pr_auc = average_precision_score(y_va, p)
            logloss_ = log_loss(y_va, p, labels=[0,1])
            brier = brier_score_loss(y_va, p)
            ks = ks_statistic(y_va.values, p)
            # observed classification metrics
            prec_obs = tp / max(tp + fp, 1e-12)
            rec_obs  = tp / max(tp + fn, 1e-12)
            acc_obs  = (tp + tn) / max(len(y_va), 1e-12)
            f1_obs   = (2 * prec_obs * rec_obs) / max(prec_obs + rec_obs, 1e-12)

            # Expected (true prevalence)
            _, pos_true, neg_true = self._natural_counts("val", int(k))
            w = class_weights_from_counts(y_va.values, pos_true, neg_true)
            exp = expected_threshold_metrics(y_va.values, p, thr, pos_true, neg_true)
            pr_auc_w = float(average_precision_score(y_va, p, sample_weight=w))
            logloss_w = weighted_logloss(y_va, p, w)
            brier_w = weighted_brier(y_va, p, w)

            # Baselines
            # Constant probability baseline at true prevalence (probability metrics)
            p_base = pos_true / max(pos_true + neg_true, 1e-12)
            base_obs = self._baseline_constant_prob(y_va.values, p_base, None); base_obs["auc"] = 0.5
            base_exp = self._baseline_constant_prob(y_va.values, p_base, w);    base_exp["auc"] = 0.5

            # Random classifier baseline: same predicted-positive rate as the model (on observed val)
            rate_pred_obs = float(pred.mean())
            base_cls_obs = random_baseline_expected_metrics(rate_pred_obs,
                                                            pos_true=int((y_va==1).sum()),
                                                            neg_true=int((y_va==0).sum()))
            base_cls_exp = random_baseline_expected_metrics(rate_pred_obs,
                                                            pos_true=pos_true, neg_true=neg_true)

            row = dict(
                label=self.label_tag, split="val", fold=int(k),
                n=len(y_va), pos=int(y_va.sum()), neg=int((y_va==0).sum()),
                # Observed model
                auc=auc, pr_auc=pr_auc, logloss=logloss_, brier=brier, ks=ks,
                precision=prec_obs, recall=rec_obs, accuracy=acc_obs, f1=f1_obs, threshold=thr,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                # Expected model
                exp_pos=pos_true, exp_neg=neg_true,
                exp_pr_auc=pr_auc_w, exp_logloss=logloss_w, exp_brier=brier_w,
                exp_tp=exp["exp_tp"], exp_fp=exp["exp_fp"], exp_tn=exp["exp_tn"], exp_fn=exp["exp_fn"],
                exp_precision=exp["exp_precision"], exp_recall=exp["exp_recall"],
                exp_accuracy=exp["exp_accuracy"], exp_f1=exp["exp_f1"],
                tpr=exp["tpr"], fpr=exp["fpr"],
                # Baseline: constant-prob (prob metrics)
                base_prob_obs_auc=base_obs["auc"], base_prob_obs_pr_auc=base_obs["pr_auc"],
                base_prob_obs_logloss=base_obs["logloss"], base_prob_obs_brier=base_obs["brier"],
                base_prob_exp_auc=base_exp["auc"], base_prob_exp_pr_auc=base_exp["pr_auc"],
                base_prob_exp_logloss=base_exp["logloss"], base_prob_exp_brier=base_exp["brier"],
                # Baseline: random classifier (classification metrics)
                base_rand_obs_tp=base_cls_obs["tp"], base_rand_obs_fp=base_cls_obs["fp"],
                base_rand_obs_tn=base_cls_obs["tn"], base_rand_obs_fn=base_cls_obs["fn"],
                base_rand_obs_precision=base_cls_obs["precision"],
                base_rand_obs_recall=base_cls_obs["recall"],
                base_rand_obs_accuracy=base_cls_obs["accuracy"],
                base_rand_obs_f1=base_cls_obs["f1"],
                base_rand_exp_tp=base_cls_exp["tp"], base_rand_exp_fp=base_cls_exp["fp"],
                base_rand_exp_tn=base_cls_exp["tn"], base_rand_exp_fn=base_cls_exp["fn"],
                base_rand_exp_precision=base_cls_exp["precision"],
                base_rand_exp_recall=base_cls_exp["recall"],
                base_rand_exp_accuracy=base_cls_exp["accuracy"],
                base_rand_exp_f1=base_cls_exp["f1"],
                base_rate_pred=rate_pred_obs, base_p_const=p_base,
            )
            metrics_rows.append(row)

        # ---------- Test metrics (calibrated probs, using global validation threshold) ----------
        X_te, y_te = prepare_xy(self.test_df, self.target_col)
        p_raw_te = self.model.predict_proba(X_te)[:, 1]
        p_te = self.calibrator.transform(p_raw_te)
        thr = self.val_thr_expected

        pred_te = (p_te >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred_te, labels=[0,1]).ravel()

        auc_te = roc_auc_score(y_te, p_te)
        pr_auc_te = average_precision_score(y_te, p_te)
        logloss_te = log_loss(y_te, p_te, labels=[0,1])
        brier_te = brier_score_loss(y_te, p_te)
        ks_te = ks_statistic(y_te.values, p_te)

        prec_te = tp / max(tp + fp, 1e-12)
        rec_te  = tp / max(tp + fn, 1e-12)
        acc_te  = (tp + tn) / max(len(y_te), 1e-12)
        f1_te   = (2 * prec_te * rec_te) / max(prec_te + rec_te, 1e-12)

        # Expected (true prevalence) on test
        _, pos_true_te, neg_true_te = self._natural_counts("test", 0)
        w_te = class_weights_from_counts(y_te.values, pos_true_te, neg_true_te)
        exp_te = expected_threshold_metrics(y_te.values, p_te, thr, pos_true_te, neg_true_te)
        pr_auc_te_w = float(average_precision_score(y_te, p_te, sample_weight=w_te))
        logloss_te_w = weighted_logloss(y_te, p_te, w_te)
        brier_te_w = weighted_brier(y_te, p_te, w_te)

        # Baselines on test
        p_base_te = pos_true_te / max(pos_true_te + neg_true_te, 1e-12)
        base_obs_te = self._baseline_constant_prob(y_te.values, p_base_te, None); base_obs_te["auc"] = 0.5
        base_exp_te = self._baseline_constant_prob(y_te.values, p_base_te, w_te);  base_exp_te["auc"] = 0.5

        rate_pred_te = float(pred_te.mean())
        base_cls_obs_te = random_baseline_expected_metrics(rate_pred_te,
                                                           pos_true=int((y_te==1).sum()),
                                                           neg_true=int((y_te==0).sum()))
        base_cls_exp_te = random_baseline_expected_metrics(rate_pred_te,
                                                           pos_true=pos_true_te, neg_true=neg_true_te)

        test_row = dict(
            label=self.label_tag, split="test", fold=0,
            n=len(y_te), pos=int(y_te.sum()), neg=int((y_te==0).sum()),
            auc=auc_te, pr_auc=pr_auc_te, logloss=logloss_te, brier=brier_te, ks=ks_te,
            precision=prec_te, recall=rec_te, accuracy=acc_te, f1=f1_te, threshold=thr,
            tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
            exp_pos=pos_true_te, exp_neg=neg_true_te,
            exp_pr_auc=pr_auc_te_w, exp_logloss=logloss_te_w, exp_brier=brier_te_w,
            exp_tp=exp_te["exp_tp"], exp_fp=exp_te["exp_fp"], exp_tn=exp_te["exp_tn"], exp_fn=exp_te["exp_fn"],
            exp_precision=exp_te["exp_precision"], exp_recall=exp_te["exp_recall"],
            exp_accuracy=exp_te["exp_accuracy"], exp_f1=exp_te["exp_f1"],
            tpr=exp_te["tpr"], fpr=exp_te["fpr"],
            # baselines:
            base_prob_obs_auc=base_obs_te["auc"], base_prob_obs_pr_auc=base_obs_te["pr_auc"],
            base_prob_obs_logloss=base_obs_te["logloss"], base_prob_obs_brier=base_obs_te["brier"],
            base_prob_exp_auc=base_exp_te["auc"], base_prob_exp_pr_auc=base_exp_te["pr_auc"],
            base_prob_exp_logloss=base_exp_te["logloss"], base_prob_exp_brier=base_exp_te["brier"],
            base_rand_obs_tp=base_cls_obs_te["tp"], base_rand_obs_fp=base_cls_obs_te["fp"],
            base_rand_obs_tn=base_cls_obs_te["tn"], base_rand_obs_fn=base_cls_obs_te["fn"],
            base_rand_obs_precision=base_cls_obs_te["precision"],
            base_rand_obs_recall=base_cls_obs_te["recall"],
            base_rand_obs_accuracy=base_cls_obs_te["accuracy"],
            base_rand_obs_f1=base_cls_obs_te["f1"],
            base_rand_exp_tp=base_cls_exp_te["tp"], base_rand_exp_fp=base_cls_exp_te["fp"],
            base_rand_exp_tn=base_cls_exp_te["tn"], base_rand_exp_fn=base_cls_exp_te["fn"],
            base_rand_exp_precision=base_cls_exp_te["precision"],
            base_rand_exp_recall=base_cls_exp_te["recall"],
            base_rand_exp_accuracy=base_cls_exp_te["accuracy"],
            base_rand_exp_f1=base_cls_exp_te["f1"],
            base_rate_pred=rate_pred_te, base_p_const=p_base_te,
        )
        metrics_rows.append(test_row)

        metrics_df = pd.DataFrame(metrics_rows).sort_values(["label","split","fold"])
        metrics_df.to_csv(self.out_dir / "metrics_per_fold_and_test.csv", index=False)

        # ===== Plots on TEST (calibrated) =====
        plot_roc(y_te.values, p_te, thr, self.out_dir / "roc_curve_test.png")
        plot_pr(y_te.values, p_te, thr, self.out_dir / "pr_curve_test.png")

        # Calibration & lift (observed + expected)
        calib = self._calibration_bins(y_te.values, p_te, bins=20)
        calib.to_csv(self.out_dir / "calibration_bins_test_observed.csv", index=False)
        plot_calibration(calib, "mean_pred", "rate",
                         self.out_dir / "calibration_test_observed.png", "Calibration (observed, test)")

        calib_w = self._calibration_bins_weighted(y_te.values, p_te, w_te, bins=20)
        calib_w.to_csv(self.out_dir / "calibration_bins_test_expected.csv", index=False)
        plot_calibration(calib_w, "mean_pred_w", "rate_w",
                         self.out_dir / "calibration_test_expected.png", "Calibration (expected, test)")

        lift = self._lift_table(y_te.values, p_te, deciles=10)
        lift.to_csv(self.out_dir / "lift_table_test_observed.csv", index=False)
        plot_lift(lift, "decile", "lift", self.out_dir / "lift_test_observed.png", "Lift by decile (observed, test)")

        lift_w = self._lift_table_weighted(y_te.values, p_te, w_te, deciles=10)
        lift_w.to_csv(self.out_dir / "lift_table_test_expected.csv", index=False)
        plot_lift(lift_w, "decile", "lift_w", self.out_dir / "lift_test_expected.png", "Lift by decile (expected, test)")

        # ---------- Confusion matrices (side-by-side with baseline) ----------
        # Observed CM: integers
        cm_obs_model = np.array([[tn, fp],[fn, tp]], dtype=float)
        # Observed random baseline CM (expected counts on the observed sample size)
        pos_s = int((y_te == 1).sum()); neg_s = int((y_te == 0).sum())
        base_obs_cm = np.array([
            [base_cls_obs_te["tn"], base_cls_obs_te["fp"]],
            [base_cls_obs_te["fn"], base_cls_obs_te["tp"]]
        ], dtype=float)
        plot_confusions_side_by_side(cm_obs_model, base_obs_cm,
                                     ("Model (observed)", "Random baseline (observed)"),
                                     self.out_dir / "confusion_test_observed_side_by_side.png",
                                     "Confusion matrices (observed, test)")

        # Expected CM: floats
        cm_exp_model = np.array([
            [exp_te["exp_tn"], exp_te["exp_fp"]],
            [exp_te["exp_fn"], exp_te["exp_tp"]]
        ], dtype=float)
        cm_exp_base = np.array([
            [base_cls_exp_te["tn"], base_cls_exp_te["fp"]],
            [base_cls_exp_te["fn"], base_cls_exp_te["tp"]]
        ], dtype=float)
        plot_confusions_side_by_side(cm_exp_model, cm_exp_base,
                                     ("Model (expected)", "Random baseline (expected)"),
                                     self.out_dir / "confusion_test_expected_side_by_side.png",
                                     "Confusion matrices (expected, test)")

        # ---------- Classification metric bars (Model vs Baseline) ----------
        model_obs_metrics = {"precision": prec_te, "recall": rec_te, "accuracy": acc_te, "f1": f1_te}
        base_obs_metrics  = {k: base_cls_obs_te[k] for k in ["precision","recall","accuracy","f1"]}
        plot_metric_bars(model_obs_metrics, base_obs_metrics,
                         "Classification metrics (observed, test)",
                         self.out_dir / "classification_bars_test_observed.png")

        model_exp_metrics = {
            "precision": exp_te["exp_precision"],
            "recall":    exp_te["exp_recall"],
            "accuracy":  exp_te["exp_accuracy"],
            "f1":        exp_te["exp_f1"],
        }
        base_exp_metrics = {
            "precision": base_cls_exp_te["precision"],
            "recall":    base_cls_exp_te["recall"],
            "accuracy":  base_cls_exp_te["accuracy"],
            "f1":        base_cls_exp_te["f1"],
        }
        plot_metric_bars(model_exp_metrics, base_exp_metrics,
                         "Classification metrics (expected, test)",
                         self.out_dir / "classification_bars_test_expected.png")

        # Score distribution + Feature importance
        plot_score_hist(y_te.values, p_te, self.out_dir / "score_hist_test.png")
        plot_feature_importance(self.model.get_booster(), self.out_dir / "feature_importance_top20.png", topn=20)

        # Summary
        summary = {
            "label": self.label_tag,
            "best_params": self.best_params,
            "threshold_chosen_on_validation_expectedF1": float(self.val_thr_expected),
            "observed_test": {
                "auc": auc_te, "pr_auc": pr_auc_te, "logloss": logloss_te, "brier": brier_te,
                "ks": ks_te, "precision": prec_te, "recall": rec_te, "accuracy": acc_te, "f1": f1_te,
                "n": int(len(y_te)), "pos": int(y_te.sum()), "neg": int((y_te==0).sum()),
                "baseline_const_prob": {
                    "auc": 0.5, "pr_auc": base_obs_te["pr_auc"], "logloss": base_obs_te["logloss"], "brier": base_obs_te["brier"]
                },
                "baseline_random_same_rate": {
                    "rate_pred": rate_pred_te,
                    "precision": base_cls_obs_te["precision"], "recall": base_cls_obs_te["recall"],
                    "f1": base_cls_obs_te["f1"], "accuracy": base_cls_obs_te["accuracy"]
                }
            },
            "expected_test": {
                "pr_auc": pr_auc_te_w, "logloss": logloss_te_w, "brier": brier_te_w,
                "f1": exp_te["exp_f1"], "precision": exp_te["exp_precision"],
                "recall": exp_te["exp_recall"], "accuracy": exp_te["exp_accuracy"],
                "tp": exp_te["exp_tp"], "fp": exp_te["exp_fp"], "tn": exp_te["exp_tn"], "fn": exp_te["exp_fn"],
                "baseline_const_prob": {
                    "auc": 0.5, "pr_auc": base_exp_te["pr_auc"], "logloss": base_exp_te["logloss"], "brier": base_exp_te["brier"]
                },
                "baseline_random_same_rate": {
                    "rate_pred": rate_pred_te,
                    "precision": base_cls_exp_te["precision"], "recall": base_cls_exp_te["recall"],
                    "f1": base_cls_exp_te["f1"], "accuracy": base_cls_exp_te["accuracy"]
                }
            },
            "run_id": self.run_id,
        }
        with open(self.out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[info] {self.label_tag}: saved metrics (observed + expected + baselines), "
              f"calibration/lift, confusion side-by-side, and bars to {self.out_dir}")

    # Convenience wrappers to reuse earlier tables for plotting
    def _calibration_bins(self, y_true: np.ndarray, y_prob: np.ndarray, bins: int = 20) -> pd.DataFrame:
        df = pd.DataFrame({"y": y_true, "p": y_prob})
        df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
        return df.groupby("bin").agg(n=("y","size"), mean_pred=("p","mean"), rate=("y","mean")).reset_index()

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
            n=("y","size"), responders=("y","sum"), avg_score=("p","mean"), rate=("y","mean")
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


# =========================
# Main
# =========================
def main():
    np.random.seed(RANDOM_STATE)
    print(f"[run] saving outputs to: {RUN_DIR.resolve()}")

    for tag in LABELS:
        print(f"\n=== Training label_tag={tag} ===")
        trainer = LabelTrainer(tag, RUN_DIR)
        print("loading latest run data...")
        trainer.load()
        print("data loaded")
        trainer.bayes_optimize()
        trainer.fit_final()
        # NEW: calibration + expected-threshold selection
        trainer.calibrate_and_choose_expected_threshold()
        trainer.evaluate_and_save()

    print("\n[done] All models trained. See run folder for artifacts.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[error]", e)
        sys.exit(1)

