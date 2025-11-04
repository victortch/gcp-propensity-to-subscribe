"""
Inference package for the Economedia Propensity-to-Subscribe engine.

Modules (added next):
    entrypoint.py   - Vertex AI Custom Job entrypoint for daily batch inference.
    batch_predict.py- Loads production model artifacts, reads features_daily,
                      outputs predictions to BigQuery.
    schema.py       - Central BigQuery schema definitions used by inference.
"""
__all__ = [
    "entrypoint",
    "batch_predict",
    "schema",
]
