"""
Training package for the Economedia Propensity-to-Subscribe engine.

Modules (added in subsequent steps):
    entrypoint.py      - Vertex AI Custom Job entrypoint for training runs.
    cv_build.py        - Builds train_data and cv_build_metadata in BigQuery from SQL.
    train.py           - Cross-validation, Bayesian opt, XGBoost fit, isotonic calibration, thresholding.
    artifact_io.py     - Persist/load artifacts to/from GCS and register in Vertex AI Model Registry.
    metrics_to_bq.py   - Write training metrics into BigQuery tables.

This package mirrors the existing local scripts (data_engineering7.py, model_training_local4.py)
without changing data processing behavior.
"""
__all__ = [
    "entrypoint",
    "cv_build",
    "train",
    "artifact_io",
    "metrics_to_bq",
]
