"""
I/O utilities shared by training and inference.

- Config:
    load_env_config(): load configs/env.yaml (local-only) with fallback to configs/env.example.yaml,
    with optional environment variable overrides.

- Google Cloud Storage (GCS):
    gcs_parse_uri(), gcs_download_bytes(), gcs_upload_bytes(),
    gcs_download_json(), gcs_upload_json(),
    gcs_download_file(), gcs_upload_file()

- BigQuery:
    get_bq_client(), get_bqstorage_client(),
    bq_query_to_df(), bq_load_dataframe()

Notes:
- In Vertex AI jobs we rely on Application Default Credentials (workload identity / SA on the job).
- Locally you can authenticate with: `gcloud auth application-default login`.
"""

from __future__ import annotations

import io
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

# BigQuery / GCS clients (lazy-import inside functions if needed)
from google.cloud import bigquery, storage
from google.cloud.bigquery import QueryJobConfig, LoadJobConfig
from google.cloud.bigquery.table import TimePartitioning
try:
    from google.cloud import bigquery_storage
    _HAS_BQSTORAGE = True
except Exception:
    bigquery_storage = None
    _HAS_BQSTORAGE = False


# =========================
# Config
# =========================

_DEF_PROJECT = "economedia-data-prod-laoy"
_DEF_LOCATION = "europe-west3"

_ENV_PATH = Path("configs/env.yaml")               # local (not committed)
_ENV_EXAMPLE_PATH = Path("configs/env.example.yaml")  # committed default


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # PyYAML
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_env_config() -> Dict[str, Any]:
    """
    Load environment config for the pipeline.

    Order of precedence:
      1) Environment variables (PROJECT_ID, REGION, etc.) – if present.
      2) configs/env.yaml – developer-local overrides (not committed).
      3) configs/env.example.yaml – repo default.

    Returns:
        dict with keys used across the project (project_id, region, bq_dataset, etc.).
    """
    cfg = {}
    if _ENV_EXAMPLE_PATH.exists():
        cfg.update(_read_yaml(_ENV_EXAMPLE_PATH))
    if _ENV_PATH.exists():
        cfg.update(_read_yaml(_ENV_PATH))

    # Allow common env var overrides (optional)
    env_overrides = {
        "project_id": os.getenv("PROJECT_ID"),
        "region": os.getenv("REGION"),
        "bq_dataset": os.getenv("BQ_DATASET"),
        "gcs_model_bucket": os.getenv("GCS_MODEL_BUCKET"),
    }
    for k, v in env_overrides.items():
        if v:
            cfg[k] = v

    # Sensible fallbacks
    cfg.setdefault("project_id", _DEF_PROJECT)
    cfg.setdefault("region", _DEF_LOCATION)
    cfg.setdefault("bq_dataset", "propensity_to_subscribe")
    return cfg


# =========================
# GCS helpers
# =========================

def gcs_parse_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    no_scheme = uri[5:]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob


def _get_storage_client() -> storage.Client:
    # ADC: works in Vertex AI and locally if you ran `gcloud auth application-default login`
    return storage.Client(project=os.getenv("PROJECT_ID", _DEF_PROJECT))


def gcs_download_bytes(uri: str) -> bytes:
    bucket_name, blob_name = gcs_parse_uri(uri)
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


def gcs_upload_bytes(uri: str, data: bytes, content_type: Optional[str] = None) -> None:
    bucket_name, blob_name = gcs_parse_uri(uri)
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(io.BytesIO(data), rewind=True, content_type=content_type)


def gcs_download_json(uri: str) -> Any:
    data = gcs_download_bytes(uri)
    return json.loads(data.decode("utf-8"))


def gcs_upload_json(uri: str, obj: Any, indent: int = 2) -> None:
    data = json.dumps(obj, indent=indent).encode("utf-8")
    gcs_upload_bytes(uri, data, content_type="application/json")


def gcs_download_file(uri: str, local_path: str | Path) -> None:
    bucket_name, blob_name = gcs_parse_uri(uri)
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))


def gcs_upload_file(uri: str, local_path: str | Path, content_type: Optional[str] = None) -> None:
    bucket_name, blob_name = gcs_parse_uri(uri)
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type=content_type)


def gcs_upload_pickle(uri: str, obj: Any) -> None:
    buf = pickle.dumps(obj)
    gcs_upload_bytes(uri, buf, content_type="application/octet-stream")


def gcs_download_pickle(uri: str) -> Any:
    buf = gcs_download_bytes(uri)
    return pickle.loads(buf)


# =========================
# BigQuery helpers
# =========================

def get_bq_client(project: Optional[str] = None, location: Optional[str] = None) -> bigquery.Client:
    """Create a BigQuery client using ADC."""
    project = project or os.getenv("PROJECT_ID", _DEF_PROJECT)
    location = location or os.getenv("BQ_LOCATION", _DEF_LOCATION)
    return bigquery.Client(project=project, location=location)


def get_bqstorage_client():
    if not _HAS_BQSTORAGE:
        return None
    try:
        return bigquery_storage.BigQueryReadClient()
    except Exception:
        return None


def bq_query_to_df(sql: str,
                   *,
                   params: Optional[Iterable[bigquery.ScalarQueryParameter]] = None,
                   project: Optional[str] = None,
                   location: Optional[str] = None) -> pd.DataFrame:
    """
    Execute SQL and return a pandas DataFrame.
    Uses BigQuery Storage API if available; falls back otherwise.
    """
    client = get_bq_client(project, location)
    job_cfg = QueryJobConfig()
    if params:
        job_cfg.query_parameters = list(params)
    job = client.query(sql, job_config=job_cfg)
    result = job.result()
    bqs = get_bqstorage_client()
    return result.to_dataframe(bqstorage_client=bqs) if bqs else result.to_dataframe()


def bq_load_dataframe(df: pd.DataFrame,
                      table_fqn: str,
                      *,
                      write_disposition: str = "WRITE_APPEND",
                      time_partitioning: Optional[TimePartitioning] = None,
                      schema: Optional[list[bigquery.SchemaField]] = None,
                      clustering_fields: Optional[list[str]] = None,
                      labels: Optional[Dict[str, str]] = None) -> bigquery.job.LoadJob:
    """
    Load a pandas DataFrame to BigQuery.

    Args:
        df: DataFrame to load.
        table_fqn: Fully-qualified table name: `project.dataset.table`.
        write_disposition: e.g., WRITE_APPEND (default), WRITE_TRUNCATE, WRITE_EMPTY.
        time_partitioning: optional TimePartitioning.
        schema: optional schema (useful when creating a new table with precise types).
        clustering_fields: optional clustering fields.
        labels: optional dict of job labels.

    Returns:
        BigQuery LoadJob (await .result() to block).
    """
    client = get_bq_client()
    job_cfg = LoadJobConfig(write_disposition=write_disposition, schema=schema)
    if time_partitioning:
        job_cfg.time_partitioning = time_partitioning
    if clustering_fields:
        job_cfg.clustering_fields = clustering_fields
    if labels:
        job_cfg.labels = labels

    job = client.load_table_from_dataframe(df, table_fqn, job_config=job_cfg)
    return job

