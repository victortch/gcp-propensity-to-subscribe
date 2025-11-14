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
from sklearn.model_selection import GroupKFold

# ------------------------
# Constants (defaults come from env in entrypoint)
# ------------------------

# These are provided by entrypoint via kwargs – defaults here are safe fallbacks
PROJECT_ID = os.getenv("PROJECT_ID", "propensity-to-subscr-eng-prod")
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
    df = df.sort_values(["user_id", "date"]).copy()
    grp = df.groupby("user_id", sort=False)

    change = (
        grp[y_col]
        .apply(lambda s: s.ne(s.shift(1)).fillna(True).astype("int8"))
        .reset_index(level=0, drop=True)
    )

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

def drop_consecutive_duplicates(df: pd.DataFrame,
                                group_col: str = "user_id",
                                order_col: str = "date",
                                exclude_cols: Optional[set] = None) -> Tuple[pd.DataFrame, int, int]:
    """
    Collapse consecutive rows per user where *all other columns* (excluding `exclude_cols`)
    are identical. Keeps the first row of each identical run. Matches legacy behavior.
    """
    if exclude_cols is None:
        exclude_cols = {order_col}
    else:
        exclude_cols = set(exclude_cols) | {order_col}

    before = len(df)
    df = df.sort_values([group_col, order_col]).copy()
    sig_cols = [c for c in df.columns if c not in exclude_cols and c != group_col]
    if sig_cols:
        sig = pd.util.hash_pandas_object(df[sig_cols], index=False)
        prev_sig = sig.groupby(df[group_col]).shift()
        keep_mask = prev_sig.isna() | sig.ne(prev_sig)
        df2 = df.loc[keep_mask].copy()
    else:
        df2 = df.copy()
    after = len(df2)
    return df2, before, after


def balance_eval_set(df: pd.DataFrame, *, random_state: int = 42) -> pd.DataFrame:
    """
    Validation/Test balancing: keep all positives; sample negatives to match #positives.
    Sampling is without replacement when possible, otherwise with replacement.
    Matches legacy behavior.
    """
    if df.empty:
        return df.copy()

    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]
    n_pos = len(pos)
    if n_pos == 0:
        # strict legacy behavior: if no positives, return empty
        return df.iloc[0:0].copy()

    replace = n_pos > len(neg)
    neg_sampled = neg.sample(n=n_pos if len(neg) > 0 else 0, replace=replace, random_state=random_state) if len(neg) > 0 else neg.iloc[0:0]
    out = pd.concat([pos, neg_sampled], axis=0).sort_values(["user_id", "date"])
    return out


def meta_drop_record(*,
                     y_col: str,
                     label_tag: str,
                     before_len: int,
                     after_len: int,
                     before_users: int,
                     after_users: int,
                     date_min: date,
                     date_max: date) -> Dict:
    """Run/label-level prep summary + constants for prep_metadata."""
    split_dt = (pd.Timestamp(PARAMS["freeze_date"]) - pd.Timedelta(days=120)).date()
    return {
        # run/label identifiers
        "run_id": RUN_ID,
        "run_started_at": RUN_STARTED_AT,
        "git_sha": GIT_SHA,
        "sql_sha": SQL_SHA,
        "label_tag": label_tag,
        "y_col": y_col,

        # prep stats
        "base_rows": int(before_len),
        "post_drop_rows": int(after_len),
        "base_users": int(before_users),
        "post_drop_users": int(after_users),
        "duplicates_dropped": int(before_len - after_len),

        # run constants
        "start_date": str(PARAMS["start_date"]),
        "freeze_date": str(PARAMS["freeze_date"]),
        "dne_start": str(PARAMS["dne_start"]),
        "cap_wall1_offer_start": str(PARAMS["cap_wall1_offer_start"]),
        "random_state": int(RANDOM_STATE),
        "test_user_frac": float(TEST_USER_FRAC),
        "n_folds": int(N_FOLDS),
        "embargo_days": int(EMBARGO_DAYS),
        "split_date": str(split_dt),

        # data extent for the label
        "date_min": str(date_min),
        "date_max": str(date_max),

        "created_at": _now_utc_iso(),
    }


def meta_split_record(*,
                      run_id: str,
                      label_tag: str,
                      split: str,
                      fold: int,
                      counts: Dict[str, int]) -> Dict:
    """Lean per-split/fold record for cv_build_metadata."""
    return {
        "run_id": run_id,
        "label_tag": label_tag,
        "split": split,                 # "train" | "val" | "test"
        "fold": int(fold),              # 0 for test
        "n": int(counts.get("n", 0)),
        "pos": int(counts.get("pos", 0)),
        "neg": int(counts.get("neg", 0)),
        "users": int(counts.get("users", 0)),
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
        bigquery.SchemaField("label_tag", "STRING"),
        bigquery.SchemaField("split", "STRING"),
        bigquery.SchemaField("fold", "INTEGER"),
        bigquery.SchemaField("n", "INTEGER"),
        bigquery.SchemaField("pos", "INTEGER"),
        bigquery.SchemaField("neg", "INTEGER"),
        bigquery.SchemaField("users", "INTEGER"),
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
    
def upload_prep_meta_to_bq(records: List[Dict[str, Any]], prep_meta_table_fqn: str, run_id: str):
    if not records:
        return
    bq = get_bq_client(PROJECT_ID, BQ_LOCATION)
    ensure_dataset(prep_meta_table_fqn.split(".")[1])
    df = pd.DataFrame.from_records(records)
    df["run_started_at"] = pd.to_datetime(df["run_started_at"], utc=True)

    schema = [
        bigquery.SchemaField("run_id", "STRING"),
        bigquery.SchemaField("run_started_at", "TIMESTAMP"),
        bigquery.SchemaField("git_sha", "STRING"),
        bigquery.SchemaField("sql_sha", "STRING"),
        bigquery.SchemaField("label_tag", "STRING"),
        bigquery.SchemaField("y_col", "STRING"),
    
        # prep stats
        bigquery.SchemaField("base_rows", "INTEGER"),
        bigquery.SchemaField("post_drop_rows", "INTEGER"),
        bigquery.SchemaField("base_users", "INTEGER"),
        bigquery.SchemaField("post_drop_users", "INTEGER"),
        bigquery.SchemaField("duplicates_dropped", "INTEGER"),
    
        # run constants
        bigquery.SchemaField("start_date", "STRING"),
        bigquery.SchemaField("freeze_date", "STRING"),
        bigquery.SchemaField("dne_start", "STRING"),
        bigquery.SchemaField("cap_wall1_offer_start", "STRING"),
        bigquery.SchemaField("random_state", "INTEGER"),
        bigquery.SchemaField("test_user_frac", "FLOAT"),
        bigquery.SchemaField("n_folds", "INTEGER"),
        bigquery.SchemaField("embargo_days", "INTEGER"),
        bigquery.SchemaField("split_date", "STRING"),
    
        # data extent for label
        bigquery.SchemaField("date_min", "STRING"),
        bigquery.SchemaField("date_max", "STRING"),
    
        bigquery.SchemaField("created_at", "STRING"),
    ]


    label_value = sanitize_for_bq_label(run_id)
    job = bq.load_table_from_dataframe(
        df, prep_meta_table_fqn,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema=schema,
            schema_update_options=[SchemaUpdateOption.ALLOW_FIELD_ADDITION],
            labels={"run_id": label_value, "dest": "prep_metadata"},
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
    
    
    
    
    # --- Legacy post-processing (matches the old script) ----------------------
    # 1) Ensure date type
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    # 2) Labels to nullable ints (same as old)
    for c in [c for c in df.columns if c.startswith("y_")]:
        df[c] = df[c].astype("Int32")
    
    # 3) RFV & components: fill NA with 0 (keep any *_missing flags from SQL intact)
    rfv_cols = [
        "recency_cap", "frequency_cap", "value_cap", "rfv_cap",
        "recency_dne", "frequency_dne", "value_dne", "rfv_dne",
    ]
    for c in rfv_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    
    # 4) Demographics: parse numeric tokens, add *_missing and *_invalid flags
    #    - *_missing: value is NA or empty string
    #    - *_invalid: non-missing but cannot be parsed as an integer
    for c in ["education", "workpos", "sex"]:
        if c in df.columns:
            s = df[c].astype("string")
            miss = s.isna() | s.str.strip().eq("")
            token = s.str.extract(r"(-?\d+)")[0]              # pull first signed integer substring
            num = pd.to_numeric(token, errors="coerce")
            df[c] = num.fillna(0).astype("Int32")            # legacy: fill 0 in numeric field
            df[f"{c}_missing"] = miss.astype("Int8")
            df[f"{c}_invalid"] = ((~miss) & num.isna()).astype("Int8")
    
    # 5) Last payment outcome to nullable int (unchanged)
    if "last_payment_outcome" in df.columns:
        df["last_payment_outcome"] = df["last_payment_outcome"].astype("Int32")
    
    # 6) Basic sanity
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
    # Legacy behavior: create y early and drop all y_* columns
    df["y"] = df[y_col].astype(int)
    df = df.drop(columns=[c for c in df.columns if c.startswith("y_")], errors="ignore")



    # Global split date: explicit freeze_date - 120 days (as in original)
    date_min = pd.to_datetime(df["date"]).min()
    date_max = pd.to_datetime(df["date"]).max()
    split_date = pd.Timestamp(PARAMS["freeze_date"]) - pd.Timedelta(days=120)
    embargo = pd.Timedelta(days=EMBARGO_DAYS)


    # compute pre-drop users on the y_col-filtered frame
    before_users = df["user_id"].nunique()
    
    before = len(df)
    df, _, after = drop_consecutive_duplicates(df, group_col="user_id", order_col="date", exclude_cols={"date"})
    after_users = df["user_id"].nunique()

    
    prep_meta = [meta_drop_record(
        y_col=y_col,
        label_tag=label_tag,
        before_len=before,
        after_len=after,
        before_users=before_users,
        after_users=after_users,
        date_min=date_min.date(),
        date_max=date_max.date(),
    )]


    # Users
    users = df["user_id"].dropna().to_numpy(dtype=np.int64, copy=True)
    users = np.unique(users)
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(users)
    

    n_test = int((TEST_USER_FRAC * len(users)))
    test_users = set(users[:n_test])
    dev_users = set(users[n_test:])

    # Natural test split (legacy: no post-cutoff; y already exists)
    te_nat = df[df["user_id"].isin(test_users)].copy()
    # y is already present; no need to touch y_* again
    te_nat_bal = balance_eval_set(te_nat, random_state=RANDOM_STATE)


    # --- metadata + folds construction (NATURAL counts in metadata; balanced rows only for uploads) ---

    dates = {"min": date_min.date(), "max": date_max.date()}
    
    # Test: NATURAL counts (use te_nat, NOT the balanced copy)
    counts_test_nat = {
        "n": len(te_nat),
        "pos": int((te_nat["y"] == 1).sum()),
        "neg": int((te_nat["y"] == 0).sum()),
        "users": te_nat["user_id"].nunique(),
    }
    meta.append(meta_split_record(run_id=RUN_ID, label_tag=label_tag, split="test", fold=0, counts=counts_test_nat))
    
    
    


    
    # Build dev/test frames first (dev for folds, test already computed above)
    dev_df = df[df["user_id"].isin(dev_users)].copy()
    
    # GroupKFold on dev users; then apply time cuts (train ≤ split, val > split)
    n_groups = max(2, dev_df["user_id"].nunique())
    gkf = GroupKFold(n_splits=min(N_FOLDS, n_groups))
    
    folds_out: List[Tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for fold_id, (_, val_idx) in enumerate(gkf.split(dev_df, groups=dev_df["user_id"]), start=1):
        va_users = set(dev_df.iloc[val_idx]["user_id"].unique())
        tr_users = set(dev_users) - va_users
    
        # NATURAL train slice for counts (users in tr_users, up to split_date - embargo)
        tr_nat_base = df[df["user_id"].isin(tr_users)].copy()
        tr_nat_base = tr_nat_base[tr_nat_base["date"] <= (split_date - embargo)]
        c_tr_nat = {
            "n": len(tr_nat_base),
            "pos": int((tr_nat_base["y"] == 1).sum()),
            "neg": int((tr_nat_base["y"] == 0).sum()),
            "users": tr_nat_base["user_id"].nunique(),
        }
    
        # NATURAL val slice for counts (users in va_users, after split_date + embargo)
        va_nat_base = df[df["user_id"].isin(va_users)].copy()
        va_nat_base = va_nat_base[va_nat_base["date"] > (split_date + embargo)]
        c_va_nat = {
            "n": len(va_nat_base),
            "pos": int((va_nat_base["y"] == 1).sum()),
            "neg": int((va_nat_base["y"] == 0).sum()),
            "users": va_nat_base["user_id"].nunique(),
        }
    
        # Record NATURAL counts to metadata
        meta.append(meta_split_record(run_id=RUN_ID, label_tag=label_tag, split="train", fold=fold_id, counts=c_tr_nat))
        meta.append(meta_split_record(run_id=RUN_ID, label_tag=label_tag, split="val",   fold=fold_id, counts=c_va_nat))
    
        # ---- Build TRAIN rows to upload: per-class one-per-streak
        tr_df_all = tr_nat_base.copy()
        # compute streak ids over y
        tr_df_all = add_streak_ids(tr_df_all, "y")  # uses 'y' column we created earlier
        # first per streak within each class
        pos = tr_df_all[tr_df_all["y"] == 1].copy()
        neg = tr_df_all[tr_df_all["y"] == 0].copy()
        pos = pos.sort_values(["user_id", "_streak_id", "date"], kind="mergesort").groupby(["user_id", "_streak_id"], as_index=False).nth(0)
        neg = neg.sort_values(["user_id", "_streak_id", "date"], kind="mergesort").groupby(["user_id", "_streak_id"], as_index=False).nth(0)
        # drop helper col
        pos = pos.drop(columns=["_streak_id"], errors="ignore")
        neg = neg.drop(columns=["_streak_id"], errors="ignore")
        
        if BALANCE_TRAINING:
            n = min(len(pos), len(neg))
            if n == 0:
                tr_df = pd.DataFrame(columns=tr_df_all.columns)
            else:
                if len(pos) > n: pos = pos.sample(n, random_state=RANDOM_STATE)
                if len(neg) > n: neg = neg.sample(n, random_state=RANDOM_STATE)
                tr_df = pd.concat([pos, neg], axis=0).sort_values(["user_id","date"]).reset_index(drop=True)
        else:
            tr_df = pd.concat([pos, neg], axis=0).sort_values(["user_id","date"]).reset_index(drop=True)

    
        # ---- Build VAL rows to upload (we'll balance at upload time per decision #1)
        va_df = va_nat_base.copy()
    
        folds_out.append((fold_id, tr_df, va_df))


    
    # Return balanced test rows for upload, processed folds for upload, and NATURAL metadata
    return te_nat_bal, folds_out, meta, prep_meta



# ------------------------
# Entrypoint function
# ------------------------

RUN_ID = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ID_LABEL = sanitize_for_bq_label(RUN_ID)
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
    run_id: str,                   # <-- NEW
    dry_run_sql: bool = False,
) -> None:
    """
    Execute the training data build:
      1) Load & run SQL to get base dataframe (identical SQL as before)
      2) Build CV splits (user-disjoint test; group folds on dev), balance as before
      3) Append slices to train_data and metadata to cv_build_metadata

    Idempotency: rows for this RUN_ID are deleted before appending.
    """
    global PROJECT_ID, BQ_LOCATION, BQ_DATASET, SQL_SHA, PARAMS, RUN_ID, RUN_ID_LABEL, RUN_STARTED_AT
    PROJECT_ID = project_id
    BQ_LOCATION = region
    BQ_DATASET = dataset

    # Set run identifiers from caller (instead of import-time default)
    RUN_ID = run_id
    RUN_ID_LABEL = sanitize_for_bq_label(RUN_ID)
    RUN_STARTED_AT = datetime.now(timezone.utc)
    
    
    prep_meta_records_all: List[Dict] = []
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

        te_nat, folds, meta, prep_meta = prepare_cv_global(df_base, y_col=y_col, label_tag=tag)

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

            va_bal = balance_eval_set(va_df.copy(), random_state=RANDOM_STATE)
            va = va_bal.copy()
            va["label"], va["split"], va["fold"] = tag, "val", fold_id
            va = tag_with_run(va, RUN_ID)
            upload_df_to_bq(va, train_fqn, RUN_ID)
            uploaded_rows_total += len(va)

            del tr, va, tr_df, va_df
            gc.collect()

        meta_records_all.extend(meta)
        prep_meta_records_all.extend(prep_meta)


    # Upload metadata
    upload_meta_to_bq(meta_records_all, meta_fqn, RUN_ID)
    prep_meta_fqn = f"{PROJECT_ID}.{BQ_DATASET}.prep_metadata"
    delete_existing_run_rows(prep_meta_fqn, RUN_ID)
    upload_prep_meta_to_bq(prep_meta_records_all, prep_meta_fqn, RUN_ID)

    logger.info("Completed RUN_ID=%s | uploaded rows: %d | meta rows: %d", RUN_ID, uploaded_rows_total, len(meta_records_all))
