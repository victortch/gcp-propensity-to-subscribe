"""
Persist training metrics to BigQuery (behavior-neutral).

This module appends compact per-label summaries to the train_metrics table.
It does NOT modify model training, data processing, or thresholds.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from google.cloud import bigquery

from app.common.io import get_bq_client
from app.common.bq_utils import create_table_if_not_exists


# BigQuery schema for propensity_to_subscribe.train_metrics
_SCHEMA: List[bigquery.SchemaField] = [
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("label_tag", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("auc_val_mean", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("threshold_expected", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("params_json", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]


def _ensure_table(project_id: str, dataset: str, table: str) -> str:
    """
    Create the metrics table if missing (idempotent) and return its FQN.
    """
    client = get_bq_client(project_id)
    table_fqn = f"{project_id}.{dataset}.{table}"
    create_table_if_not_exists(
        project=project_id,
        dataset_id=dataset,
        table_id=table,
        schema=_SCHEMA,
        partition_field=None,          # simple append-only table
        clustering_fields=None,
        description="Economedia PTS training metrics (one row per label and run).",
        labels={"purpose": "pts-train-metrics"},
    )
    return table_fqn


def write_training_summary(
    *,
    project_id: str,
    dataset: str,
    table: str,
    run_id: str,
    label_tag: str,
    auc_val_mean: float,
    threshold_expected: float,
    params: Dict[str, Any],
) -> None:
    """
    Append a single summary row to the train_metrics table.

    Args:
        project_id, dataset, table: BigQuery destination identifiers.
        run_id: training run identifier.
        label_tag: e.g., "cap_30d".
        auc_val_mean: mean validation ROC AUC across folds.
        threshold_expected: chosen global threshold (expected F1).
        params: best hyperparameters dict (stored as JSON string).
    """
    table_fqn = _ensure_table(project_id, dataset, table)
    client = get_bq_client(project_id)

    row = {
        "run_id": run_id,
        "label_tag": label_tag,
        "auc_val_mean": float(auc_val_mean),
        "threshold_expected": float(threshold_expected),
        "params_json": pd.io.json.dumps(params, ensure_ascii=False, sort_keys=True),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    df = pd.DataFrame([row])

    job = client.load_table_from_dataframe(
        df,
        table_fqn,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
        ),
    )
    job.result()