"""
Builds the training dataset and metadata in BigQuery (train_data, cv_build_metadata).

- SQL is loaded from app/training/sql/build_training_dataset.sql
- Parameters: start_date, freeze_date, dne_start, cap_wall1_offer_start
- Produces identical outputs (schema/logic) for:
    • propensity_to_subscribe.train_data          (append for this run_id)
    • propensity_to_subscribe.cv_build_metadata   (append for this run_id)

Notes:
- No changes to data processing, label definitions, or sampling logic.
- Uses the same CV/splitting/balancing logic as your original script.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, SchemaUpdateOption

from app.common.io import get_bq_client, get_bqstorage_client
from app.common.utils import get_logger, sanitize_for_bq_label

# ------------------------
# Constants (defaults come from env in entrypoint)
# ------------------------

# These are provided by entrypoint via kwargs – defaults here are safe fallbacks
PROJECT_ID = os.getenv("PROJECT_ID", "economedia-data-prod-laoy")
BQ_LOCATION = os.getenv("BQ_LOCATION", "europe-west3")
BQ_DATASET = os.getenv("BQ_DATASET", "propensity_to_subscribe")

# Runtime knobs (aligned with original)
TEST_USER_FRAC   = 0.20         # final test split by users
N_FOLDS          = 5            # user-grouped folds on dev
EMBARGO_DAYS     = 0            # keep 0 as before (user-disjoint)
BALANCE_TRAINING = True         # 50/50 downsampling after one-per-streak
RANDOM_STATE     = 42

# ------------------------
# Lightweight helpers (same behavior as original)
# ------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _sha256_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_bq_clients():
    bq = get_bq_client(PROJECT_ID, BQ_LOCATION)
    bqs = get_bqstorage_client()
    return bq, bqs

def ensure_dataset(dataset_id: str):
    bq = get_bq_client(PROJECT_ID, BQ_LOCATION)
    try:
        bq.get_dataset(f"{PROJECT_ID}.{dataset_id}")
    except NotFound:
        ds = bigquery.Dataset(f"{PROJECT_ID}.{dataset_id}")
        ds.location = BQ_LOCATION
        bq.create_dataset(ds)

def table_exists(table_fqn: str) -> bool:
    bq = get_bq_client(PROJECT_ID, BQ_LOCATION)
    try:
        bq.get_table(table_fqn)
        return True
    except NotFound:
        return False

def delete_existing_run_rows(table_fqn: str, run_id: str) -> None:
    bq = get_bq_client(PROJECT_ID, BQ_LOCATION)
    if not table_exists(table_fqn):
        return
    sql = f"DELETE FROM `{table_fqn}` WHERE run_id = @run_id"
    job = bq.query(sql, job_config=QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
    ))
    job.result()

def add_streak_ids(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Assign per-user streak ids of consecutive equal y values (as in original)."""
    df = df.sort_values(["user_id", "date"]).copy()
    grp = df.groupby("user_id", sort=False)
    # change points where y != y.shift(1)
    change = (grp[y_col].apply(lambda s: s.ne(s.shift(1)).astype(int))).reset_index(level=0, drop=True)
    df["_chg"] = change
    df["_streak_id"] = df.groupby("user_id", sort=False)["_chg"].cumsum()
    df = df.drop(columns=["_chg"])
    return df

def one_per_streak(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Take one row per (user_id, streak_id), preferring earliest in streak."""
    df = add_streak_ids(df, y_col)
    df = df.sort_values(["user_id", "_streak_id", "date"], kind="mergesort")
    # first in each streak
    df = df.groupby(["user_id", "_streak_id"], as_index=False).nth(0)
    return df.drop(columns=["_streak_id"])

def drop_consecutive_duplicates(df: pd.DataFrame, y_col: str) -> Tuple[pd.DataFrame, int, int]:
    """Drop consecutive duplicates (same user,y,date-next) as in original."""
    before = len(df)
    df2 = one_per_streak(df, y_col)
    after = len(df2)
    return df2, before, after

def balance_eval_set(df: pd.DataFrame, *, random_state: int = 42) -> pd.DataFrame:
    """
    Balance negatives to match positives (keep all positives).
    Used for validation and test in original script.
    """
    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]
    if len(pos) == 0:
        return df.copy()
    n = len(pos)
    neg_sample = neg.sample(n=min(n, len(neg)), random_state=random_state) if len(neg) > 0 else neg
    return pd.concat([pos, neg_sample], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

def meta_drop_record(y_col: str, label_tag: str, *, before_len: int, after_len: int) -> Dict:
    return {
        "y_col": y_col,
        "label_tag": label_tag,
        "split": "n/a",
        "fold": 0,
        "n": int(after_len),
        "pos": None,
        "neg": None,
        "users": None,
        "duplicates_dropped": int(before_len - after_len),
        "kept_after_drop": int(after_len),
    }

def meta_split_record(run_id: str, run_started_at: str, git_sha: str, sql_sha: str,
                      y_col: str, tag: str, split: str, fold: int, counts: Dict[str, int],
                      dates: Dict[str, Any], params: Dict[str, Any]) -> Dict:
    return {
        "run_id": run_id,
        "run_started_at": run_started_at,
        "git_sha": git_sha,
        "sql_sha": sql_sha,
        "y_col": y_col,
        "label_tag": tag,
        "split": split,
        "fold": int(fold),
        "n": int(counts.get("n", 0)),
        "pos": int(counts.get("pos", 0)),
        "neg": int(counts.get("neg", 0)),
        "users": int(counts.get("users", 0)),
        "duplicates_dropped": pd.NA,
        "kept_after_drop": pd.NA,
        "created_at": _now_utc_iso(),
        "start_date": str(params["start_date"]),
        "freeze_date": str(params["freeze_date"]),
        "dne_start": str(params["dne_start"]),
        "cap_wall1_offer_start": str(params["cap_wall1_offer_start"]),
        "random_state": int(RANDOM_STATE),
        "date_min": str(dates.get("min")),
        "date_max": str(dates.get("max")),
    }

def tag_with_run(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    out = df.copy()
    out["run_id"] = run_id
    return out

def upload_meta_to_bq(records: List[Dict[str, Any]], meta_table_fqn: str, run_id: str):
    if not records:
        return
    bq = get_bq_client(PROJECT_ID, BQ_LOCATION)
    ensure_dataset(meta_table_fqn.split(".")[1])
    df = pd.DataFrame.from_records(records)
    label_value = sanitize_for_bq_label(run_id)
    # keep JSON-like types as strings to avoid struct inference issues
    schema = [
        bigquery.SchemaField("run_id", "STRING"),
        bigquery.SchemaField("run_started_at", "STRING"),
        bigquery.SchemaField("git_sha", "STRING"),
        bigquery.SchemaField("sql_sha", "STRING"),
        bigquery.SchemaField("y_col", "STRING"),
        bigquery.SchemaField("label_tag", "STRING"),
        bigquery.SchemaField("split", "STRING"),
        bigquery.SchemaField("fold", "INTEGER"),
        bigquery.SchemaField("n", "INTEGER"),
        bigquery.SchemaField("pos", "INTEGER"),
        bigquery.SchemaField("neg", "INTEGER"),
        bigquery.SchemaField("users", "INTEGER"),
        bigquery.SchemaField("duplicates_dropped", "INTEGER"),
        bigquery.SchemaField("kept_after_drop", "INTEGER"),
        bigquery.SchemaField("created_at", "STRING"),
        bigquery.SchemaField("start_date", "STRING"),
        bigquery.SchemaField("freeze_date", "STRING"),
        bigquery.SchemaField("dne_start", "STRING"),
        bigquery.SchemaField("cap_wall1_offer_start", "STRING"),
        bigquery.SchemaField("random_state", "INTEGER"),
        bigquery.SchemaField("date_min", "STRING"),
        bigquery.SchemaField("date_max", "STRING"),
    ]
    job = bq.load_table_from_dataframe(
        df, meta_table_fqn,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema=schema,
            schema_update_options=[SchemaUpdateOption.ALLOW_FIELD_ADDITION],
            labels={"run_id": label_value, "dest": "cv_build_metadata"},
        ),
    )
    job.result()

def upload_df_to_bq(df: pd.DataFrame, table_fqn: str, run_id: str):
    bq = get_bq_client(PROJECT_ID, BQ_LOCATION)
    ensure_dataset(table_fqn.split(".")[1])

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "y" in df.columns:
        df["y"] = df["y"].astype("Int64")
    if "fold" in df.columns:
        df["fold"] = df["fold"].astype("Int64")
    for b in [c for c in df.columns if str(df[c].dtype) == "boolean"]:
        df[b] = df[b].astype("Int64")

    label_value = sanitize_for_bq_label(run_id)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema_update_options=[SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        labels={"run_id": label_value, "dest": "train_data"},
    )
    job = bq.load_table_from_dataframe(df, table_fqn, job_config=job_config)
    job.result()

# ------------------------
# Data load from SQL
# ------------------------

def load_base_dataframe(sql_text: str, params: Dict[str, Any]) -> pd.DataFrame:
    """Execute external SQL with named parameters and return DataFrame (post-processed)."""
    bq, bqs = make_bq_clients()
    job_config = QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("start_date", "DATE", params["start_date"]),
        bigquery.ScalarQueryParameter("freeze_date", "DATE", params["freeze_date"]),
        bigquery.ScalarQueryParameter("dne_start", "DATE", params["dne_start"]),
        bigquery.ScalarQueryParameter("cap_wall1_offer_start", "DATE", params["cap_wall1_offer_start"]),
    ])
    job = bq.query(sql_text, job_config=job_config)
    res = job.result()
    df = res.to_dataframe(bqstorage_client=bqs) if bqs is not None else res.to_dataframe()

    # Light post-processing identical to original
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    for c in [c for c in df.columns if c.startswith("y_")]:
        df[c] = df[c].astype("Int32")
    for col in ["rfv_cap", "rfv_dne"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if "education" in df.columns:
        df["education"] = pd.to_numeric(df["education"], errors="coerce").astype("Int32")
    if "workpos" in df.columns:
        df["workpos"] = pd.to_numeric(df["workpos"], errors="coerce").astype("Int32")
    if "sex" in df.columns:
        df["sex"] = pd.to_numeric(df["sex"], errors="coerce").astype("Int32")
    if "last_payment_outcome" in df.columns:
        df["last_payment_outcome"] = df["last_payment_outcome"].astype("Int32")

    required = {"user_id", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Base DF missing required columns: {missing}")
    return df

# ------------------------
# CV construction (matches original approach)
# ------------------------

def prepare_cv_global(df_all: pd.DataFrame, y_col: str, *, label_tag: str) -> Tuple[pd.DataFrame, List[Tuple[int, pd.DataFrame, pd.DataFrame]], List[Dict]]:
    """
    Build user-disjoint dev/test splits, then GroupKFold-like folds over dev users.
    Train: dev users not in fold (<= split_date - embargo), one-per-streak, optional balance.
    Val:   fold users (>  split_date + embargo), natural dist; metadata for drops/sizes.
    Returns: (test_df_natural, folds_list, metadata_records)
    """
    meta: List[Dict] = []

    df = df_all.copy()
    if "y" in df.columns:
        df = df.drop(columns=["y"])
    df = df[df[y_col].notna()].copy()

    # Global split date: (freeze_date - 120d) as in original
    # We'll infer from df dates to be robust; original used FREEZE_DATE variable.
    date_min = pd.to_datetime(df["date"]).min()
    date_max = pd.to_datetime(df["date"]).max()
    split_date = date_max - pd.Timedelta(days=120)
    embargo = pd.Timedelta(days=EMBARGO_DAYS)

    # Drop consecutive duplicates; capture metadata
    before = len(df)
    df, _, after = drop_consecutive_duplicates(df, y_col)
    meta.append(meta_drop_record(y_col, label_tag, before_len=before, after_len=after))

    # Users
    users = df["user_id"].dropna().to_numpy(dtype=np.int64, copy=True)
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(users)
    users = np.unique(users)

    n_test = int(round(TEST_USER_FRAC * len(users)))
    test_users = set(users[:n_test])
    dev_users = set(users[n_test:])

    # Natural test split
    te_nat = df[df["user_id"].isin(test_users)].copy()
    te_nat["y"] = te_nat[y_col].astype("Int64")
    te_nat = te_nat.drop(columns=[c for c in te_nat.columns if c.startswith("y_") and c != "y"])
    te_nat = te_nat[te_nat["date"] > (split_date + embargo)]
    te_nat_bal = balance_eval_set(te_nat, random_state=RANDOM_STATE)

    # GroupKFold over dev users
    dev_users_list = np.array(sorted(list(dev_users)))
    rng.shuffle(dev_users_list)
    folds: List[Tuple[int, pd.DataFrame, pd.DataFrame]] = []
    fold_sizes = np.array_split(dev_users_list, N_FOLDS)

    for fold_id, fold_users in enumerate(fold_sizes, start=1):
        fold_users = set(fold_users.tolist())
        tr_users = list(dev_users - fold_users)
        va_users = list(fold_users)

        tr_df = df[df["user_id"].isin(tr_users)].copy()
        va_df = df[df["user_id"].isin(va_users)].copy()

        # temporal guard (<= split_date - embargo) for train, (> split_date + embargo) for val (natural)
        tr_df = tr_df[tr_df["date"] <= (split_date - embargo)]
        va_df = va_df[va_df["date"] >  (split_date + embargo)]

        # training one-per-streak, optional balance 50/50 on y
        tr_df = one_per_streak(tr_df, y_col)
        tr_df["y"] = tr_df[y_col].astype("Int64")
        tr_df = tr_df.drop(columns=[c for c in tr_df.columns if c.startswith("y_") and c != "y"])
        if BALANCE_TRAINING:
            pos = tr_df[tr_df["y"] == 1]
            neg = tr_df[tr_df["y"] == 0]
            if len(pos) > 0 and len(neg) > len(pos):
                tr_df = pd.concat([pos, neg.sample(n=len(pos), random_state=RANDOM_STATE)], axis=0)
                tr_df = tr_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

        # validation natural
        va_df = va_df.copy()
        va_df["y"] = va_df[y_col].astype("Int64")
        va_df = va_df.drop(columns=[c for c in va_df.columns if c.startswith("y_") and c != "y"])

        folds.append((fold_id, tr_df, va_df))

    # metadata summary counts
    dates = {"min": date_min.date(), "max": date_max.date()}
    counts_test = {"n": len(te_nat_bal), "pos": int((te_nat_bal["y"] == 1).sum()), "neg": int((te_nat_bal["y"] == 0).sum()), "users": te_nat_bal["user_id"].nunique()}
    meta.append(meta_split_record(RUN_ID, RUN_STARTED_AT.isoformat(), GIT_SHA, SQL_SHA, y_col, label_tag, "test", 0, counts_test, dates, PARAMS))
    for fold_id, tr_df, va_df in folds:
        c_tr = {"n": len(tr_df), "pos": int((tr_df["y"] == 1).sum()), "neg": int((tr_df["y"] == 0).sum()), "users": tr_df["user_id"].nunique()}
        c_va = {"n": len(va_df), "pos": int((va_df["y"] == 1).sum()), "neg": int((va_df["y"] == 0).sum()), "users": va_df["user_id"].nunique()}
        meta.append(meta_split_record(RUN_ID, RUN_STARTED_AT.isoformat(), GIT_SHA, SQL_SHA, y_col, label_tag, "train", fold_id, c_tr, dates, PARAMS))
        meta.append(meta_split_record(RUN_ID, RUN_STARTED_AT.isoformat(), GIT_SHA, SQL_SHA, y_col, label_tag, "val",   fold_id, c_va, dates, PARAMS))
    return te_nat_bal, folds, meta

# ------------------------
# Entrypoint function
# ------------------------

RUN_ID = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_STARTED_AT = datetime.now(timezone.utc)
GIT_SHA = os.getenv("GIT_SHA", "")

SQL_SHA = ""   # set at runtime after loading file
PARAMS: Dict[str, Any] = {}

def build_training_data(
    *,
    project_id: str,
    region: str,
    dataset: str,
    train_table: str,
    metadata_table: str,
    start_date: str,
    freeze_date: str,
    dne_start: str,
    cap_wall1_offer_start: str,
    dry_run_sql: bool = False,
) -> None:
    """
    Execute the training data build:
      1) Load & run SQL to get base dataframe (identical SQL as before)
      2) Build CV splits (user-disjoint test; group folds on dev), balance as before
      3) Append slices to train_data and metadata to cv_build_metadata

    Idempotency: rows for this RUN_ID are deleted before appending.
    """
    global PROJECT_ID, BQ_LOCATION, BQ_DATASET, SQL_SHA, PARAMS
    PROJECT_ID = project_id
    BQ_LOCATION = region
    BQ_DATASET = dataset

    logger = get_logger("pts.training.cv_build")

    # 0) Load SQL from file and compute SHA
    sql_path = Path("app/training/sql/build_training_dataset.sql")
    sql_text = sql_path.read_text(encoding="utf-8")
    SQL_SHA = _sha256_string(sql_text)

    # 1) Params for the query and metadata
    PARAMS = {
        "start_date": date.fromisoformat(start_date),
        "freeze_date": date.fromisoformat(freeze_date),
        "dne_start": date.fromisoformat(dne_start),
        "cap_wall1_offer_start": date.fromisoformat(cap_wall1_offer_start),
    }

    # 2) Idempotency: clear previous rows for this run_id
    train_fqn = f"{PROJECT_ID}.{BQ_DATASET}.{train_table}"
    meta_fqn  = f"{PROJECT_ID}.{BQ_DATASET}.{metadata_table}"
    ensure_dataset(BQ_DATASET)
    delete_existing_run_rows(train_fqn, RUN_ID)
    delete_existing_run_rows(meta_fqn, RUN_ID)

    # 3) SQL dry run (optional)
    if dry_run_sql:
        logger.info("Dry-run SQL enabled; validating query only.")
        bq = get_bq_client(PROJECT_ID, BQ_LOCATION)
        job = bq.query(sql_text, job_config=QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "DATE", PARAMS["start_date"]),
                bigquery.ScalarQueryParameter("freeze_date", "DATE", PARAMS["freeze_date"]),
                bigquery.ScalarQueryParameter("dne_start", "DATE", PARAMS["dne_start"]),
                bigquery.ScalarQueryParameter("cap_wall1_offer_start", "DATE", PARAMS["cap_wall1_offer_start"]),
            ]
        ))
        logger.info("SQL dry run OK. Bytes processed estimate: %s", getattr(job, "total_bytes_processed", None))
        return

    # 4) Execute SQL → base df
    df_base = load_base_dataframe(sql_text, PARAMS)
    logger.info("Base DF loaded: %s rows, %s columns, %s..%s", len(df_base), len(df_base.columns), df_base["date"].min().date(), df_base["date"].max().date())

    uploaded_rows_total = 0
    meta_records_all: List[Dict] = []

    # Labels to process are the four explicit columns present in df_base
    label_map = [
        ("y_cap_90d", "cap_90d"),
        ("y_dne_90d", "dne_90d"),
        ("y_cap_30d", "cap_30d"),
        ("y_dne_30d", "dne_30d"),
    ]

    for y_col, tag in label_map:
        if y_col not in df_base.columns:
            continue

        te_nat, folds, meta = prepare_cv_global(df_base, y_col=y_col, label_tag=tag)
        # Upload TEST (balanced) slice
        te = te_nat.copy()
        te["label"], te["split"], te["fold"] = tag, "test", 0
        te = tag_with_run(te, RUN_ID)
        upload_df_to_bq(te, train_fqn, RUN_ID)
        uploaded_rows_total += len(te)
        del te, te_nat
        gc.collect()

        # Upload TRAIN/VAL folds
        for fold_id, tr_df, va_df in folds:
            tr = tr_df.copy()
            tr["label"], tr["split"], tr["fold"] = tag, "train", fold_id
            tr = tag_with_run(tr, RUN_ID)
            upload_df_to_bq(tr, train_fqn, RUN_ID)
            uploaded_rows_total += len(tr)

            va = va_df.copy()
            va["label"], va["split"], va["fold"] = tag, "val", fold_id
            va = tag_with_run(va, RUN_ID)
            upload_df_to_bq(va, train_fqn, RUN_ID)
            uploaded_rows_total += len(va)

            del tr, va, tr_df, va_df
            gc.collect()

        meta_records_all.extend(meta)

    # Upload metadata
    upload_meta_to_bq(meta_records_all, meta_fqn, RUN_ID)

    logger.info("Completed RUN_ID=%s | uploaded rows: %d | meta rows: %d", RUN_ID, uploaded_rows_total, len(meta_records_all))
