"""
Vertex AI Custom Job entrypoint for training the Economedia PTS model.

This script only orchestrates:
  1) Build training dataset in BigQuery (train_data, cv_build_metadata)
  2) Run model training, calibration, threshold selection, metrics persistence, and model registration

It does NOT change feature processing or modeling behavior; those are implemented
to match the existing local scripts inside cv_build.py and train.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Any, Optional

from app.common.io import load_env_config
from app.common.utils import get_logger, make_run_id, utcnow_iso
# Implemented in upcoming steps:
# - app/training/cv_build.py
# - app/training/train.py
from app.training import cv_build, train


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTS Training Entrypoint")

    # Data window parameters (forwarded to cv_build; required for reproducibility)
    p.add_argument("--start_date", required=True, help="YYYY-MM-DD; historical features/labels start")
    p.add_argument("--freeze_date", required=True, help="YYYY-MM-DD; last date included in training window")
    p.add_argument("--dne_start", required=True, help="YYYY-MM-DD; Dnevnik paywall/offer start, used by SQL")
    p.add_argument("--cap_wall1_offer_start", required=True, help="YYYY-MM-DD; Capital wall1/offer start, used by SQL")

    # Optional knobs
    p.add_argument("--primary_label", default=None, help="Default/primary label for summaries (e.g., cap_30d)")
    p.add_argument("--labels_yaml", default="configs/labels.yaml", help="Path to labels.yaml")
    p.add_argument("--thresholds_policy", default="configs/thresholds_policy.yaml", help="Path to thresholds_policy.yaml")
    p.add_argument("--log_level", default=None, help="Override log level (INFO, DEBUG, etc.)")

    # Advanced / debug
    p.add_argument("--dry_run_sql", action="store_true", help="If set, only validates SQL without writing tables")
    p.add_argument("--run_id", default=None, help="Override run id; if omitted a new one is generated")


    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg: Dict[str, Any] = load_env_config()

    logger = get_logger("pts.training.entrypoint", level=args.log_level or cfg.get("log_level", "INFO"))
    run_id = args.run_id or make_run_id()

    logger.info("=== PTS Training Entrypoint ===")
    logger.info("Run ID: %s  |  Timestamp: %s", run_id, utcnow_iso())

    # Echo key config for traceability (safe, no secrets in example file)
    safe_cfg = {
        "project_id": cfg.get("project_id"),
        "region": cfg.get("region"),
        "bq_dataset": cfg.get("bq_dataset"),
        "bq_train_data_table": cfg.get("bq_train_data_table", "train_data"),
        "bq_cv_metadata_table": cfg.get("bq_cv_metadata_table", "cv_build_metadata"),
        "bq_train_metrics_table": cfg.get("bq_train_metrics_table", "train_metrics"),
        "gcs_model_bucket": cfg.get("gcs_model_bucket"),
        "vertex_model_display_name": cfg.get("vertex_model_display_name", "pts_xgb_model"),
    }
    logger.info("Config: %s", json.dumps(safe_cfg, indent=2))

    # 1) Build / refresh the training dataset in BigQuery
    logger.info("Step 1/2: Building training data in BigQuery...")

    cv_params = {
        "project_id": cfg["project_id"],
        "region": cfg["region"],
        "dataset": cfg["bq_dataset"],
        "train_table": cfg.get("bq_train_data_table", "train_data"),
        "metadata_table": cfg.get("bq_cv_metadata_table", "cv_build_metadata"),
        "start_date": args.start_date,
        "freeze_date": args.freeze_date,
        "dne_start": args.dne_start,
        "cap_wall1_offer_start": args.cap_wall1_offer_start,
        "run_id": run_id,                      # <-- pass in the generated run_id
        "dry_run_sql": bool(args.dry_run_sql),
    }
    logger.info("cv_build params: %s", json.dumps({k: v for k, v in cv_params.items() if k != 'dry_run_sql'}, indent=2))

    # This function is implemented in a later step; here we just call it.
    cv_build.build_training_data(**cv_params)
    logger.info("Training dataset build complete.")

    # 2) Train the model (CV, Bayes opt, XGBoost, isotonic calibration, threshold selection, metrics)
    logger.info("Step 2/2: Running model training...")
    train_params = {
        "project_id": cfg["project_id"],
        "region": cfg["region"],
        "dataset": cfg["bq_dataset"],
        "train_table": cfg.get("bq_train_data_table", "train_data"),
        "metrics_table": cfg.get("bq_train_metrics_table", "train_metrics"),
        "labels_yaml": args.labels_yaml,
        "thresholds_policy": args.thresholds_policy,
        "gcs_model_bucket": cfg["gcs_model_bucket"],
        "vertex_model_display_name": cfg.get("vertex_model_display_name", "pts_xgb_model"),
        "vertex_model_registry_label": cfg.get("vertex_model_registry_label", "propensity_to_subscribe"),
        "artifact_repo": cfg.get("artifact_repo"),
        "run_id": run_id,                      # <-- same run_id gets used in training
        "primary_label": args.primary_label,
    }
    logger.info("train params: %s", json.dumps({k: v for k, v in train_params.items() if k != 'artifact_repo'}, indent=2))

    # This function is implemented in a later step; here we just call it.
    result = train.run_training(**train_params)
    # Expected `result` to contain pointers to artifacts, metrics summary, and model resource name (later step).
    logger.info("Training completed. Summary: %s", json.dumps(result or {}, indent=2))

    logger.info("=== Training run finished successfully. run_id=%s ===", run_id)


if __name__ == "__main__":
    main(sys.argv[1:])