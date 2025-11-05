"""BigQuery convenience helpers for the Economedia PTS project.

Lightweight wrappers for dataset/table existence checks, creation, and
schema serialization. Used by training and inference modules to ensure
required datasets/tables exist before reading or writing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from google.cloud import bigquery
from google.api_core.exceptions import Conflict, NotFound

from app.common.io import get_bq_client


# ============================================================
# Dataset helpers
# ============================================================

def dataset_exists(project: str, dataset_id: str) -> bool:
    client = get_bq_client(project)
    try:
        client.get_dataset(f"{project}.{dataset_id}")
        return True
    except NotFound:
        return False


def create_dataset_if_not_exists(
    project: str,
    dataset_id: str,
    location: str = "europe-west3",
    description: str | None = None,
    labels: Optional[Dict[str, str]] = None,
) -> bigquery.Dataset:
    """
    Create a dataset if it doesn't exist (idempotent).

    Returns:
        bigquery.Dataset
    """
    client = get_bq_client(project, location)
    dataset_ref = f"{project}.{dataset_id}"
    try:
        return client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        if description:
            dataset.description = description
        if labels:
            dataset.labels = labels
        try:
            dataset = client.create_dataset(dataset)
            print(f"Created dataset {dataset_ref}")
        except Conflict:
            dataset = client.get_dataset(dataset_ref)
        return dataset


# ============================================================
# Table helpers
# ============================================================

def table_exists(project: str, dataset_id: str, table_id: str) -> bool:
    client = get_bq_client(project)
    table_ref = f"{project}.{dataset_id}.{table_id}"
    try:
        client.get_table(table_ref)
        return True
    except NotFound:
        return False


def create_table_if_not_exists(
    project: str,
    dataset_id: str,
    table_id: str,
    schema: List[bigquery.SchemaField],
    partition_field: Optional[str] = None,
    clustering_fields: Optional[List[str]] = None,
    description: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
) -> bigquery.Table:
    """
    Create a table if it does not exist (idempotent).
    Useful for inference (predictions_daily) and metrics tables.
    """
    client = get_bq_client(project)
    table_ref = f"{project}.{dataset_id}.{table_id}"

    if table_exists(project, dataset_id, table_id):
        return client.get_table(table_ref)

    table = bigquery.Table(table_ref, schema=schema)
    if partition_field:
        table.time_partitioning = bigquery.TimePartitioning(field=partition_field)
    if clustering_fields:
        table.clustering_fields = clustering_fields
    if description:
        table.description = description
    if labels:
        table.labels = labels

    try:
        table = client.create_table(table)
        print(f"Created table {table_ref}")
    except Conflict:
        table = client.get_table(table_ref)
    return table


# ============================================================
# Schema helpers
# ============================================================

def schema_from_json(json_path: str) -> List[bigquery.SchemaField]:
    """
    Load a BigQuery schema from a JSON file (list of {name, type, mode}).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        fields = json.load(f)
    return [bigquery.SchemaField(**fld) for fld in fields]


def schema_to_json(schema: List[bigquery.SchemaField], out_path: str) -> None:
    """
    Dump a BigQuery schema to a JSON file.
    """
    fields = [{"name": f.name, "type": f.field_type, "mode": f.mode} for f in schema]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2)
