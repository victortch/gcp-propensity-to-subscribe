"""
Model training for Economedia PTS.

Key behaviors preserved:
- Latest run_id from cv_build_metadata.
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
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support

from google.cloud import bigquery

from app.common.io import get_bq_client
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
PROJECT_ID = os.getenv("PROJECT_ID", "economedia-data-prod-laoy")
BQ_LOCATION = os.getenv("BQ_LOCATION", "europe-west3")
BQ_DATASET = os.getenv("BQ_DATASET", "propensity_to_subscribe")

TABLE_DATA = f"{PROJECT_ID}.{BQ_DATASET}.train_data"
TABLE_META = f"{PROJECT_ID}.{BQ_DATASET}.cv_build_metadata"

RANDOM_STATE = 42
N_BAYES_INIT = 8
N_BAYES_ITER = 24


# -----------------------
# Helpers reading from BQ
# -----------------------

def latest_run_id(client: bigquery.Client) -> str:
    sql = f"""
    SELECT run_id
    FROM `{TABLE_META}`
    ORDER BY created_at DESC
    LIMIT 1
    """
    rows = list(client.query(sql).result())
    if not rows:
        raise RuntimeError("No run_id found in cv_build_metadata.")
    return rows[0]["run_id"]


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

def class_weights_from_counts(y_val: pd.Series, pos_true: int, neg_true: int) -> np.ndarray:
    """Weight each example to reflect expected positives/negatives."""
    pos_obs = max(int((y_val == 1).sum()), 1)
    neg_obs = max(int((y_val == 0).sum()), 1)
    w_pos = pos_true / pos_obs
    w_neg = neg_true / neg_obs
    w = np.where(y_val.values == 1, w_pos, w_neg)
    return w


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
# Trainer
# -----------------------

@dataclass
class TrainOutputs:
    label: str
    best_params: Dict[str, Any]
    threshold_expected: float
    auc_val_mean: float
    artifact_local_dir: str  # local path; artifact_io will upload this folder


class LabelTrainer:
    def __init__(self, label_tag: str, out_dir: Path):
        self.label_tag = label_tag
        self.target_col = "y"
        self.out_dir = out_dir / label_tag
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.client = get_bq_client(project=PROJECT_ID, location=BQ_LOCATION)
        self.run_id = latest_run_id(self.client)

        self.folds: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        self.test_df: Optional[pd.DataFrame] = None
        self.meta_counts: Optional[pd.DataFrame] = None

        self.best_params: Optional[Dict[str, Any]] = None
        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.val_thr_expected: Optional[float] = None

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

    # ---------- bayesian optimization ----------

    def _cv_score(self, max_depth, gamma, colsample_bytree, learning_rate, n_estimators, reg_alpha, reg_lambda) -> float:
        """Objective to maximize: mean ROC AUC across folds."""
        params = dict(
            objective="binary:logistic",
            eval_metric="auc",
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
            n_jobs=0,
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
        # persist locally (artifact_io will upload later)
        (self.out_dir / "best_params.json").write_text(json.dumps(self.best_params, indent=2))

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
            eval_metric="auc",
            tree_method="hist",
            subsample=1.0,
            random_state=RANDOM_STATE,
            n_jobs=0,
            **(self.best_params or {}),
        )
        self.model = xgb.XGBClassifier(**params, scale_pos_weight=spw)
        self.model.fit(X_all, y_all)
        # Save local model; artifact_io will upload later
        self.model.save_model(str(self.out_dir / f"model_{self.label_tag}.json"))

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
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_val_all, y_val_all, sample_weight=w_val)
        self.calibrator = iso
        # Persist local calibrator; artifact_io will upload later
        try:
            import joblib
            joblib.dump(iso, self.out_dir / f"isotonic_calibrator_{self.label_tag}.joblib")
        except Exception:
            pass

        # Calibrated probabilities for threshold selection
        p_val_cal = iso.transform(p_val_all)

        # Choose threshold maximizing EXPECTED F1 on validation
        thr, best_exp_f1 = best_expected_f1_threshold_global(
            y_true=y_val_all.values,
            y_prob=p_val_cal,
            pos_true=pos_true_total,
            neg_true=neg_true_total,
            grid=400,
        )
        self.val_thr_expected = float(thr)
        # Persist threshold
        (self.out_dir / f"threshold_expected_{self.label_tag}.txt").write_text(f"{self.val_thr_expected:.6f}\n")

    # ---------- evaluation helpers ----------

    def _natural_totals(self, split: str) -> Tuple[int, int]:
        """Return total expected positives/negatives across folds for a split."""
        m = self.meta_counts
        if m is None:
            raise RuntimeError("meta_counts not loaded.")
        m2 = m[m["split"] == split]
        pos_true = int(m2["pos"].sum())
        neg_true = int(m2["neg"].sum())
        return pos_true, neg_true

    @staticmethod
    def _scale_pos_weight(y: pd.Series) -> float:
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        return (neg / max(pos, 1.0))

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
            artifact_local_dir=str(self.out_dir),
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
    global PROJECT_ID, BQ_LOCATION, BQ_DATASET, TABLE_DATA, TABLE_META
    PROJECT_ID = project_id
    BQ_LOCATION = region
    BQ_DATASET = dataset
    TABLE_DATA = f"{PROJECT_ID}.{BQ_DATASET}.{train_table}"
    TABLE_META = f"{PROJECT_ID}.{BQ_DATASET}.cv_build_metadata"

    logger = get_logger("pts.training.train")

    # Load labels config
    with open(labels_yaml, "r", encoding="utf-8") as f:
        import yaml
        labels_cfg = yaml.safe_load(f) or {}
    labels = [l["id"] for l in labels_cfg.get("labels", []) if l.get("enabled", True)]
    if primary_label and primary_label not in labels:
        labels.insert(0, primary_label)

    out_root = Path("runs") / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {"run_id": run_id, "labels": {}}

    for tag in labels:
        logger.info("=== Training label: %s ===", tag)
        trainer = LabelTrainer(label_tag=tag, out_dir=out_root)
        trainer.load()
        logger.info("Bayesian optimization starting...")
        trainer.bayes_optimize()
        logger.info("Fitting final model + calibration + threshold selection...")
        trainer.fit_final()
        summary = trainer.summary()
        results["labels"][tag] = {
            "best_params": summary.best_params,
            "threshold_expected": summary.threshold_expected,
            "auc_val_mean": summary.auc_val_mean,
            "artifact_local_dir": summary.artifact_local_dir,
        }

        # Optional: if artifact_io is present (added in a later step), upload artifacts and register model
        if artifact_io is not None:
            try:
                artifact_io.save_and_register_label_run(
                    label_tag=tag,
                    local_dir=summary.artifact_local_dir,
                    gcs_model_bucket=gcs_model_bucket,
                    vertex_model_display_name=vertex_model_display_name,
                    vertex_model_registry_label=vertex_model_registry_label,
                    run_id=run_id,
                )
            except Exception as e:
                logger.warning("artifact_io.save_and_register_label_run failed: %s", e)

        # Optional: write metrics to BQ if available (later step)
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
            except Exception as e:
                logger.warning("metrics_to_bq.write_training_summary failed: %s", e)

    logger.info("Training summary: %s", json.dumps(results, indent=2))
    return results
