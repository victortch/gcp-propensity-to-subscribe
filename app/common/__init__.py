"""
Common utilities and shared logic for the Economedia Propensity-to-Subscribe engine.

Modules:
    io.py              - Handles input/output operations to BigQuery, GCS, and local files.
    bq_utils.py        - Convenience helpers for BigQuery table creation and schema management.
    preprocessing.py   - Shared feature preprocessing and transformation utilities.
    registry.py        - Interactions with Vertex AI Model Registry.
    utils.py           - Miscellaneous utilities shared by training and inference code.
"""

__all__ = [
    "io",
    "bq_utils",
    "preprocessing",
    "registry",
    "utils",
]
