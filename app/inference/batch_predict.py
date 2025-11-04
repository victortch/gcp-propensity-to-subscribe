"""
Daily batch inference for the Economedia PTS engine.

Behavior preserved:
- Feature prep mirrors the training's drop/cast rules (no one-hot): drop id/meta & y_*,
  cast numerics to float32, non-numerics -> categorical codes -> float32, fillna(0.0).
- Per-label scoring uses the XGBoost model trained in Step 17 and isotonic calibrator
  (if present), then applies the stored/overridden threshold to produce a decision.

Inputs:
- BigQuery: propensity_to_subscribe.features_daily (one row per user for scoring_date)
- Artifacts in GCS under a Vertex AI Model Registry "production" version's artifact_uri

Outputs:
- BigQuery: propensity_to_subscribe.predictions_daily (one row per user per label)

This module does not alter training logic or data processing semantics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from google.cloud import bigquery

from app.common.io import (
    get_bq_client,
    gcs_download_json,
    gcs_download_file,
    load_env_config,
)
from app.common.registry import (
    init_ai,
    resolve_production_version,
    get_artifact_uri,
)
from app.common.utils import get_logger


# ---------------------------------------------------------------------
# Small helpers (match training feature prep without labels)
# ---------------------------------------------------------------------

NON_FEATURE_COLS_BASE = {"user_id", "date", "split", "fold", "label", "run_id"}

def _prep_features_like_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors training's prepare_xy_compat(), minus target handling:
      - Drop id/meta + anything ending with '_id' and all y_* columns
      - Numeric -> float32
      - Non-numeric -> categorical codes -> float32
      - Fill NaNs with 0.0
    """
    dyn_ids = {c for c in df.columns if c.lower().endswith("_id")}
    drop_cols = set(NON_FEATURE_COLS_BASE) | dyn_ids | {c for c in df.columns if c.startswith("y_")}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
        else:
            X[c] = X[c].astype("category").cat.codes.astype("float32")
    X = X.fillna(0.0)
    return X


def _align_to_feature_names(X: pd.DataFrame, feature_names: Optional[List[str]]) -> pd.DataFrame:
    """
    If the XGBoost booster exposes feature_names, reorder/select columns accordingly.
    Otherwise return X as-is.
    """
    if not feature_names:
        return X
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing[:10]}{'...' if len(missing)>10 else ''}")
    return X[feature_names]


# ---------------------------------------------------------------------
# Artifact loading (production version) per label
# ---------------------------------------------------------------------

@dataclass
class LabelArtifacts:
    label: str
    model: xgb.XGBClassifier
    calibrator: Optional[IsotonicRegression]
    threshold: float
    model_version_name: str
    artifact_uri: str


def _load_label_artifacts(display_name: str, label_tag: str) -> LabelArtifacts:
    """
    Resolve production version, read its manifest.json, load model json,
    optional isotonic calibrator, and threshold value.
    """
    cfg = load_env_config()
    project_id = cfg["project_id"]
    region = cfg["region"]

    init_ai(project_id, region)

    m = resolve_production_version(display_name)
    if m is None:
        raise RuntimeError(f"No production model found for display_name={display_name}")
    artifact_uri = get_artifact_uri(m)
    if not artifact_uri:
        raise RuntimeError("Production model has no artifact_uri set.")

    # Manifests are per-label under artifact_uri/<label>/manifest.json
    # But artifact_uri itself may already be .../<label>. If so, read directly.
    if artifact_uri.rstrip("/").endswith(f"/{label_tag}"):
        label_prefix = artifact_uri.rstrip("/")
    else:
        label_prefix = f"{artifact_uri.rstrip('/')}/{label_tag}"

    manifest_uri = f"{label_prefix}/manifest.json"
    manifest = gcs_download_json(manifest_uri)

    # Paths for this label; fall back to conventional names
    files = manifest.get("files", {})
    model_uri = files.get("model") or f"{label_prefix}/model_{label_tag}.json"
    calib_uri = files.get("calibrator") or f"{label_prefix}/isotonic_calibrator_{label_tag}.joblib"
    thr = manifest.get("threshold_expected", None)
    if thr is None:
        # final fallback if manifest lacked it
        thr = 0.5
    threshold = float(thr)

    # Download model json to temp file and load
    tmp_dir = Path(".local_infer_cache") / label_tag
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_model = tmp_dir / f"model_{label_tag}.json"
    gcs_download_file(model_uri, local_model)
    model = xgb.XGBClassifier()
    model.load_model(str(local_model))

    # Optional calibrator
    calibrator: Optional[IsotonicRegression] = None
    try:
        from google.cloud.storage import Blob  # ensure client exists
        local_cal = tmp_dir / f"isotonic_calibrator_{label_tag}.joblib"
        try:
            gcs_download_file(calib_uri, local_cal)
            import joblib
            calibrator = joblib.load(local_cal)
        except Exception:
            calibrator = None
    except Exception:
        calibrator = None

    return LabelArtifacts(
        label=label_tag,
        model=model,
        calibrator=calibrator,
        threshold=threshold,
        model_version_name=m.resource_name,
        artifact_uri=artifact_uri,
    )


# ---------------------------------------------------------------------
# BigQuery I/O
# ---------------------------------------------------------------------

def _read_features_for_date(
    client: bigquery.Client,
    *,
    project_id: str,
    dataset: str,
    table: str,
    scoring_date: str,
) -> pd.DataFrame:
    sql = f"""
    SELECT *
    FROM `{project_id}.{dataset}.{table}`
    WHERE scoring_date = @scoring_date
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("scoring_date", "DATE", scoring_date)]
        ),
    )
    return job.result().to_dataframe()


def _write_predictions(
    client: bigquery.Client,
    *,
    project_id: str,
    dataset: str,
    table: str,
    df: pd.DataFrame,
) -> None:
    table_fqn = f"{project_id}.{dataset}.{table}"
    job = client.load_table_from_dataframe(
        df,
        table_fqn,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    )
    job.result()


# ---------------------------------------------------------------------
# Inference per label
# ---------------------------------------------------------------------

def _score_one_label(
    *,
    feats_df: pd.DataFrame,
    user_cols: List[str],
    label_art: LabelArtifacts,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      user_id, scoring_date, label, prob, decision, model_version, artifact_uri, created_at
    """
    # Preserve user/date columns for merge
    meta = feats_df[user_cols].copy()

    # Build X like training
    X = _prep_features_like_training(feats_df)

    # Align to training feature order if model exposes feature names
    try:
        booster = label_art.model.get_booster()
        feat_names = booster.feature_names
    except Exception:
        feat_names = None
    X = _align_to_feature_names(X, feat_names)

    # Raw model proba
    prob = label_art.model.predict_proba(X)[:, 1]
    # Calibrate if available
    if label_art.calibrator is not None:
        prob = label_art.calibrator.transform(prob)

    # Decision by threshold
    decision = (prob >= label_art.threshold).astype("int64")

    out = meta.copy()
    out["label"] = label_art.label
    out["prob"] = prob.astype("float64")
    out["decision"] = decision
    out["model_version"] = label_art.model_version_name
    out["artifact_uri"] = label_art.artifact_uri
    out["created_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_inference(
    *,
    project_id: str,
    region: str,
    dataset: str,
    features_table: str,
    predictions_table: str,
    gcs_model_bucket: str,  # not directly needed here but kept for parity with training params
    vertex_model_display_name: str,
    scoring_date: str,
    labels_yaml: str,
    thresholds_policy: str,
) -> Dict[str, Any]:
    """
    Run daily batch prediction for the given scoring_date.

    Steps:
      1) Load enabled labels from labels.yaml.
      2) Resolve production model → load per-label artifacts (model, calibrator, threshold).
      3) Read features for scoring_date from BigQuery.
      4) Score per label, concatenate, and write to predictions_daily.
    """
    logger = get_logger("pts.inference.batch_predict")

    # 1) Labels
    with open(labels_yaml, "r", encoding="utf-8") as f:
        import yaml
        labels_cfg = yaml.safe_load(f) or {}
    labels = [l["id"] for l in labels_cfg.get("labels", []) if l.get("enabled", True)]
    if not labels:
        raise RuntimeError("No enabled labels found in labels.yaml")

    # 2) Resolve production model & per-label artifacts
    arts: Dict[str, LabelArtifacts] = {}
    for tag in labels:
        arts[tag] = _load_label_artifacts(vertex_model_display_name, tag)

    # 3) Read features
    client = get_bq_client(project_id, region)
    feats = _read_features_for_date(
        client,
        project_id=project_id,
        dataset=dataset,
        table=features_table,
        scoring_date=scoring_date,
    )
    if feats.empty:
        logger.warning("No feature rows found for scoring_date=%s; writing nothing.", scoring_date)
        return {"status": "no_features", "scoring_date": scoring_date}

    # Keep user/date for output; accept either 'scoring_date' or 'date' in features
    date_col = "scoring_date" if "scoring_date" in feats.columns else "date"
    user_cols = ["user_id", date_col]
    missing = [c for c in user_cols if c not in feats.columns]
    if missing:
        raise ValueError(f"Features table missing required columns: {missing}")

    # 4) Score per label
    outs = []
    for tag in labels:
        out = _score_one_label(
            feats_df=feats,
            user_cols=user_cols,
            label_art=arts[tag],
        )
        outs.append(out)

    pred_df = pd.concat(outs, axis=0, ignore_index=True)

    # Normalize schema and column names for BQ
    pred_df = pred_df.rename(columns={date_col: "scoring_date"})
    # Explicit types
    pred_df["user_id"] = pd.to_numeric(pred_df["user_id"], errors="coerce").astype("Int64")
    pred_df["decision"] = pred_df["decision"].astype("Int64")
    pred_df["scoring_date"] = pd.to_datetime(pred_df["scoring_date"]).dt.date

    # 5) Write to BigQuery
    _write_predictions(
        client,
        project_id=project_id,
        dataset=dataset,
        table=predictions_table,
        df=pred_df[[
            "scoring_date", "user_id", "label", "prob", "decision",
            "model_version", "artifact_uri", "created_at"
        ]],
    )

    return {
        "status": "ok",
        "scoring_date": scoring_date,
        "labels_scored": labels,
        "rows_written": int(len(pred_df)),
    }
