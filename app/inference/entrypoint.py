"""
Vertex AI Custom Job entrypoint for daily batch inference.

What this does:
  - Loads repo/env config (no secrets).
  - Parses CLI args (e.g., --scoring_date=YYYY-MM-DD).
  - Calls app.inference.batch_predict.run_inference(...) to:
      * resolve the production model in Vertex AI Model Registry
      * load artifacts from GCS
      * read features from propensity_to_subscribe.features_daily for scoring_date
      * write calibrated probabilities + binary decisions into propensity_to_subscribe.predictions_daily

No data-processing or model math is defined here; it's purely orchestration.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from app.common.io import load_env_config
from app.common.utils import get_logger, utcnow_iso

# Implemented in the next step
from app.inference import batch_predict


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTS Inference Entrypoint")

    p.add_argument("--scoring_date", required=True, help="YYYY-MM-DD date to score (one row per user for this date).")

    # Optional knobs
    p.add_argument("--labels_yaml", default="configs/labels.yaml", help="Path to labels.yaml")
    p.add_argument("--thresholds_policy", default="configs/thresholds_policy.yaml", help="Path to thresholds_policy.yaml")
    p.add_argument("--log_level", default=None, help="Override log level (INFO, DEBUG, etc.)")

    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg: Dict[str, Any] = load_env_config()

    logger = get_logger("pts.inference.entrypoint", level=args.log_level or cfg.get("log_level", "INFO"))
    logger.info("=== PTS Inference Entrypoint ===")
    logger.info("Timestamp: %s", utcnow_iso())

    # Echo key config for traceability (safe subset)
    safe_cfg = {
        "project_id": cfg.get("project_id"),
        "region": cfg.get("region"),
        "bq_dataset": cfg.get("bq_dataset"),
        "bq_features_table": cfg.get("bq_features_table", "features_daily"),
        "bq_predictions_table": cfg.get("bq_predictions_table", "predictions_daily"),
        "gcs_model_bucket": cfg.get("gcs_model_bucket"),
        "vertex_model_display_name": cfg.get("vertex_model_display_name", "pts_xgb_model"),
    }
    logger.info("Config: %s", json.dumps(safe_cfg, indent=2))

    # Kick off batch inference (implementation in batch_predict.py)
    result = batch_predict.run_inference(
        project_id=cfg["project_id"],
        region=cfg["region"],
        dataset=cfg["bq_dataset"],
        features_table=cfg.get("bq_features_table", "features_daily"),
        predictions_table=cfg.get("bq_predictions_table", "predictions_daily"),
        gcs_model_bucket=cfg["gcs_model_bucket"],
        vertex_model_display_name=cfg.get("vertex_model_display_name", "pts_xgb_model"),
        scoring_date=args.scoring_date,
        labels_yaml=args.labels_yaml,
        thresholds_policy=args.thresholds_policy,
    )

    logger.info("Inference completed. Summary: %s", json.dumps(result or {}, indent=2))
    logger.info("=== Inference run finished successfully for %s ===", args.scoring_date)


if __name__ == "__main__":
    main(sys.argv[1:])
