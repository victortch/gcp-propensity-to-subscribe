"""Batch inference logic for the Economedia PTS project.

- Feature preprocessing mirrors training (cast numerics to ``float32``,
  categorical codes for non-numerics, fill ``NaN`` with ``0.0``).
- Per-label scoring uses the XGBoost model trained offline and optional isotonic
  calibrator, then applies the stored/overridden threshold to produce decisions.

Inputs:
- BigQuery ``propensity_to_subscribe.features_daily`` (one row per user for ``scoring_date``)
- Artifacts in GCS under the Vertex AI Model Registry production version ``artifact_uri``

Outputs:
- BigQuery ``propensity_to_subscribe.predictions_daily`` (one row per user per label)

This module does not alter training logic or data processing semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from google.cloud import bigquery

from app.common.io import (
    get_bq_client,
    gcs_download_json,
    gcs_download_file,
    gcs_upload_json,
    load_env_config,
)
from app.common.registry import (
    init_ai,
    resolve_production_version_for_label,
    get_artifact_uri,
)
from app.common.preprocessing import drop_meta_cols
from app.common.utils import get_logger

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------
# Small helpers (match training feature prep without labels)
# ---------------------------------------------------------------------

def _prep_features_like_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror training's prepare_xy_compat and cv_build.load_base_dataframe post-processing:
      - Drop meta cols + 'scoring_date'
      - If demographics flags are missing, synthesize them exactly like training:
            * parse first integer token from the string
            * fill NA with 0
            * add *_missing / *_invalid flags
      - Cast numerics to float32; non-numerics to categorical codes -> float32; fillna 0.0
    """
    dfx = df.copy()

    # Synthesize demographic features like training if they are not already present.
    # Training logic reference: app/training/cv_build.py load_base_dataframe() step 4.  <-- keeps parity
    for c in ["education", "workpos", "sex"]:
        if c in dfx.columns:
            miss_col = f"{c}_missing"
            inv_col  = f"{c}_invalid"
            # Only add if not present (idempotent and backward-compatible with future BQ updates)
            if miss_col not in dfx.columns or inv_col not in dfx.columns:
                s = dfx[c].astype("string")
                miss = s.isna() | s.str.strip().eq("")
                token = s.str.extract(r"(-?\d+)")[0]
                num = pd.to_numeric(token, errors="coerce")
                # Match training fill and dtype; downstream we cast to float32 anyway
                dfx[c] = num.fillna(0).astype("Int64")
                dfx[miss_col] = miss.astype("Int8")
                dfx[inv_col]  = ((~miss) & num.isna()).astype("Int8")

    # Drop meta cols; exclude 'scoring_date' explicitly (training uses 'date', which is dropped)
    X = drop_meta_cols(dfx, extra_drop=["scoring_date"])

    # Cast per training rules
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
        else:
            X[c] = X[c].astype("category").cat.codes.astype("float32")

    return X.fillna(0.0)


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
    threshold_source: str
    model_version_name: str
    artifact_uri: str
    run_id: Optional[str]
    feature_names: Optional[List[str]]
    manifest: Dict[str, Any]


def _parse_thresholds_policy(path: str) -> Tuple[float, Dict[str, float]]:
    """Return fallback threshold and per-label overrides."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to read thresholds_policy.yaml")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    default_cfg = cfg.get("default", {})
    fallback = float(default_cfg.get("fallback_threshold", 0.5))

    overrides: Dict[str, float] = {}
    for label, body in (cfg.get("labels") or {}).items():
        override = (body or {}).get("override") or {}
        fixed = override.get("fixed_threshold")
        if fixed is not None:
            overrides[label] = float(fixed)

    return fallback, overrides


def _load_label_artifacts(
    display_name: str,
    label_tag: str,
    *,
    threshold_override: Optional[float],
    fallback_threshold: float,
) -> LabelArtifacts:
    """
    Resolve the production version for this label, read its manifest.json,
    load the persisted model, optional calibrator, threshold, and feature names.
    """
    cfg = load_env_config()
    project_id = cfg["project_id"]
    region = cfg["region"]

    init_ai(project_id, region)

    model_version = resolve_production_version_for_label(display_name, label_tag)
    if model_version is None:
        raise RuntimeError(
            f"No production version found for display_name={display_name} label={label_tag}"
        )
    artifact_uri = get_artifact_uri(model_version)
    if not artifact_uri:
        raise RuntimeError("Selected model version has no artifact_uri set.")

    model_version_name = model_version.resource_name
    run_id_label = (getattr(model_version, "labels", {}) or {}).get("run_id")

    # Normalize to the label-specific folder (artifact_uri may already point to it)
    if artifact_uri.rstrip("/").endswith(f"/{label_tag}"):
        label_prefix = artifact_uri.rstrip("/")
    else:
        label_prefix = f"{artifact_uri.rstrip('/')}/{label_tag}"

    manifest_uri = f"{label_prefix}/manifest.json"
    manifest = gcs_download_json(manifest_uri)

    files = manifest.get("files", {}) or {}

    # Model bytes (UBJ) -> local cache -> load into XGBClassifier
    model_uri = files.get("model") or f"{label_prefix}/model_{label_tag}.ubj"
    tmp_dir = Path(".local_infer_cache") / label_tag
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_model = tmp_dir / Path(model_uri).name
    gcs_download_file(model_uri, local_model)
    model = xgb.XGBClassifier()
    model.load_model(str(local_model))

    # Optional calibrator
    calibrator: Optional[IsotonicRegression] = None
    calib_uri = files.get("calibrator") or f"{label_prefix}/isotonic_calibrator_{label_tag}.joblib"
    try:
        local_cal = tmp_dir / Path(calib_uri).name
        gcs_download_file(calib_uri, local_cal)
        import joblib

        calibrator = joblib.load(local_cal)
    except Exception:
        calibrator = None

    # Threshold resolution: override > manifest value > threshold file > fallback
    threshold: Optional[float] = None
    threshold_source = "manifest"
    if threshold_override is not None:
        threshold = float(threshold_override)
        threshold_source = "override"
    else:
        if "threshold_expected" in manifest:
            threshold = float(manifest["threshold_expected"])
        else:
            threshold_uri = files.get("threshold") or f"{label_prefix}/threshold_expected_{label_tag}.txt"
            try:
                local_thr = tmp_dir / Path(threshold_uri).name
                gcs_download_file(threshold_uri, local_thr)
                with open(local_thr, "r", encoding="utf-8") as f:
                    threshold = float((f.read() or "0.5").strip())
                threshold_source = "file"
            except Exception:
                threshold = None
        if threshold is None:
            threshold = float(fallback_threshold)
            threshold_source = "fallback"

    # Feature names ensure strict column ordering (optional)
    feature_names: Optional[List[str]]
    fn_uri = files.get("feature_names") or f"{label_prefix}/feature_names.json"
    try:
        loaded = gcs_download_json(fn_uri)
        feature_names = list(loaded or [])
    except Exception:
        feature_names = None

    return LabelArtifacts(
        label=label_tag,
        model=model,
        calibrator=calibrator,
        threshold=float(threshold),
        threshold_source=threshold_source,
        model_version_name=model_version_name,
        artifact_uri=label_prefix,
        run_id=manifest.get("run_id") or run_id_label,
        feature_names=feature_names,
        manifest=manifest,
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


def _write_inference_metadata(
    *,
    gcs_model_bucket: str,
    scoring_date: str,
    artifacts: Dict[str, LabelArtifacts],
    rows_written: int,
    status: str,
) -> Tuple[str, Dict[str, Any]]:
    """Persist a JSON sidecar summarizing the inference run."""
    now = datetime.utcnow()
    created_at = now.isoformat(timespec="seconds") + "Z"
    payload = {
        "scoring_date": scoring_date,
        "created_at": created_at,
        "status": status,
        "rows_written": int(rows_written),
        "labels": {},
    }
    for tag, art in artifacts.items():
        payload["labels"][tag] = {
            "model_version": art.model_version_name,
            "artifact_uri": art.artifact_uri,
            "threshold": float(art.threshold),
            "threshold_source": art.threshold_source,
            "run_id": art.run_id,
        }

    slug = now.strftime("%Y%m%dT%H%M%SZ")
    uri = f"{gcs_model_bucket.rstrip('/')}/inference_runs/{scoring_date}/{slug}.json"
    gcs_upload_json(uri, payload, indent=2)
    return uri, payload


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

    # Align to training feature order using persisted feature_names (preferred)
    feat_names = label_art.feature_names
    if not feat_names:
        try:
            feat_names = label_art.model.get_booster().feature_names
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
    out["threshold"] = float(label_art.threshold)
    out["model_version"] = label_art.model_version_name
    out["artifact_uri"] = label_art.artifact_uri
    out["model_run_id"] = label_art.run_id
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

    if not gcs_model_bucket:
        raise ValueError("gcs_model_bucket must be provided")

    # 1) Labels
    with open(labels_yaml, "r", encoding="utf-8") as f:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read labels.yaml")
        labels_cfg = yaml.safe_load(f) or {}
    labels = [l["id"] for l in labels_cfg.get("labels", []) if l.get("enabled", True)]
    if not labels:
        raise RuntimeError("No enabled labels found in labels.yaml")

    fallback_threshold, threshold_overrides = _parse_thresholds_policy(thresholds_policy)

    # 2) Resolve production model & per-label artifacts
    arts: Dict[str, LabelArtifacts] = {}
    for tag in labels:
        arts[tag] = _load_label_artifacts(
            vertex_model_display_name,
            tag,
            threshold_override=threshold_overrides.get(tag),
            fallback_threshold=fallback_threshold,
        )

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
        metadata_uri, _ = _write_inference_metadata(
            gcs_model_bucket=gcs_model_bucket,
            scoring_date=scoring_date,
            artifacts=arts,
            rows_written=0,
            status="no_features",
        )
        return {
            "status": "no_features",
            "scoring_date": scoring_date,
            "labels_scored": labels,
            "rows_written": 0,
            "metadata_uri": metadata_uri,
        }

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
    pred_df["threshold"] = pd.to_numeric(pred_df["threshold"], errors="coerce").astype("float64")
    pred_df["model_run_id"] = pred_df["model_run_id"].astype("string")
    pred_df["model_run_id"] = pred_df["model_run_id"].replace({pd.NA: None})

    # 5) Write to BigQuery
    _write_predictions(
        client,
        project_id=project_id,
        dataset=dataset,
        table=predictions_table,
        df=pred_df[[
            "scoring_date", "user_id", "label", "prob", "decision",
            "threshold", "model_version", "artifact_uri", "model_run_id", "created_at"
        ]],
    )

    metadata_uri, metadata_payload = _write_inference_metadata(
        gcs_model_bucket=gcs_model_bucket,
        scoring_date=scoring_date,
        artifacts=arts,
        rows_written=len(pred_df),
        status="ok",
    )

    return {
        "status": "ok",
        "scoring_date": scoring_date,
        "labels_scored": labels,
        "rows_written": int(len(pred_df)),
        "metadata_uri": metadata_uri,
        "metadata": metadata_payload,
    }
