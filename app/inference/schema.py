"""
BigQuery schema definitions for inference-time tables.

This module is behavior-neutral. It centralizes schema objects used by
inference to make table creation / validation consistent.
"""

from __future__ import annotations

from typing import List
from google.cloud import bigquery

# ---------------------------------------------------------------------
# predictions_daily (date-partitioned by scoring_date)
# Columns written by app.inference.batch_predict
# ---------------------------------------------------------------------
PREDICTIONS_DAILY_SCHEMA: List[bigquery.SchemaField] = [
    bigquery.SchemaField("scoring_date", "DATE", mode="REQUIRED"),   # partition column
    bigquery.SchemaField("user_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("label", "STRING", mode="REQUIRED"),        # e.g., cap_30d
    bigquery.SchemaField("prob", "FLOAT", mode="REQUIRED"),          # calibrated probability
    bigquery.SchemaField("decision", "INT64", mode="REQUIRED"),      # 0/1 using selected threshold
    bigquery.SchemaField("threshold", "FLOAT", mode="NULLABLE"),      # threshold applied for decision
    bigquery.SchemaField("model_version", "STRING", mode="NULLABLE"),# Vertex AI model resource name
    bigquery.SchemaField("artifact_uri", "STRING", mode="NULLABLE"), # GCS folder with artifacts
    bigquery.SchemaField("model_run_id", "STRING", mode="NULLABLE"),  # training run that produced artifacts
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),# ingestion timestamp (UTC ISO)
]

# ---------------------------------------------------------------------
# Optional: features_daily reference (for validation or DDL creation)
# We plan to create the features table via code (we use a scheduled query),
# so we keep this as a minimal reference. It won't be used to transform data.
# ---------------------------------------------------------------------
FEATURES_DAILY_MIN_SCHEMA: List[bigquery.SchemaField] = [
    bigquery.SchemaField("scoring_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("user_id", "INT64", mode="REQUIRED"),
    # ... feature columns are produced by the scheduled query; not enumerated here
]
