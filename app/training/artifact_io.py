"""
Artifact I/O utilities for training outputs.

This module:
- Uploads per-label artifacts from local run folder to GCS.
- Writes a manifest.json with pointers and small metadata.
- Registers a model version in Vertex AI Model Registry, tagged as 'candidate'.

It does NOT alter any training or preprocessing logic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.common.io import (
    gcs_upload_file,
    gcs_upload_json,
)
from app.common.registry import (
    init_ai,
    register_model_version,
)
from app.common.io import load_env_config
from app.common.utils import sanitize_for_bq_label


def _gcs_join(prefix: str, filename: str) -> str:
    prefix = prefix.rstrip("/")
    return f"{prefix}/{filename}"


def _read_text_or_none(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _read_json_or_none(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_label_files(local_dir: Path, label_tag: str) -> Dict[str, Path]:
    """
    Return a dict of expected files if present.
    """
    return {
        "model_joblib": local_dir / f"model_{label_tag}.joblib",
        "calibrator": local_dir / f"isotonic_calibrator_{label_tag}.joblib",
        "best_params": local_dir / "best_params.json",
        "threshold": local_dir / f"threshold_expected_{label_tag}.txt",
    }


def save_and_register_label_run(
    *,
    label_tag: str,
    local_dir: str,
    gcs_model_bucket: str,
    vertex_model_display_name: str,
    vertex_model_registry_label: str,
    run_id: str,
) -> Dict[str, str]:
    """
    Upload artifacts for a single label and register a 'candidate' model version.

    Args:
        label_tag: e.g., "cap_30d"
        local_dir: path created by train.py for this label (contains model_*.joblib etc.)
        gcs_model_bucket: e.g., "gs://economedia-pts-models"
        vertex_model_display_name: e.g., "pts_xgb_model"
        vertex_model_registry_label: e.g., "propensity_to_subscribe" (used as a label key/value)
        run_id: training run identifier (folder name in GCS)

    Returns:
        dict with {"artifact_uri": "gs://.../runs/<run_id>/<label>/", "model_resource_name": "..."}
    """
    cfg = load_env_config()
    project_id = cfg.get("project_id")
    region = cfg.get("region")

    # Initialize Vertex AI SDK (no-ops locally if unauthenticated)
    init_ai(project_id, region)

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local artifact dir missing: {local_dir}")

    gcs_prefix = f"{gcs_model_bucket.rstrip('/')}/runs/{run_id}/{label_tag}"
    files = _collect_label_files(local_dir, label_tag)

    # Upload known artifacts if they exist
    uploaded: Dict[str, str] = {}
    for key, path in files.items():
        if path.exists():
            target = _gcs_join(gcs_prefix, path.name)
            gcs_upload_file(target, path)
            uploaded[key] = target

    # Build manifest with small metadata
    manifest = {
        "run_id": run_id,
        "label": label_tag,
        "files": uploaded,
        "display_name": vertex_model_display_name,
        "registry_group": vertex_model_registry_label,
    }

    # Add best params + threshold values into manifest (if present)
    best_params = _read_json_or_none(files["best_params"])
    thr_text = _read_text_or_none(files["threshold"])
    if best_params is not None:
        manifest["best_params"] = best_params
    if thr_text is not None:
        try:
            manifest["threshold_expected"] = float(thr_text)
        except Exception:
            manifest["threshold_expected"] = thr_text  # keep raw if parse fails

    # Write manifest.json
    manifest_uri = _gcs_join(gcs_prefix, "manifest.json")
    gcs_upload_json(manifest_uri, manifest, indent=2)

    # Register into Vertex AI Model Registry
    run_id_label = sanitize_for_bq_label(run_id)
    labels = {
        "stage": "candidate",
        "run_id": run_id_label,
        "label": label_tag,
        "group": vertex_model_registry_label,  # discovery label
    }
    metadata = {}
    if "threshold_expected" in manifest:
        metadata["threshold_expected"] = str(manifest["threshold_expected"])
    if "best_params" in manifest:
        # Keep as JSON string for compact metadata
        metadata["best_params_json"] = json.dumps(manifest["best_params"], separators=(",", ":"))

    model = register_model_version(
        display_name=vertex_model_display_name,
        artifact_uri=gcs_prefix,
        labels=labels,
        metadata=metadata,
        version_aliases=["candidate"],
    )

    return {
        "artifact_uri": gcs_prefix,
        "model_resource_name": model.resource_name,
    }


def write_manifest_and_register_existing(
    *,
    label_tag: str,
    run_id: str,
    gcs_model_bucket: str,
    vertex_model_display_name: str,
    vertex_model_registry_label: str,
    best_params: Optional[dict],
    threshold: Optional[float],
) -> Dict[str, str]:
    """
    Write a manifest.json (no local files needed) and register a version in Vertex AI.

    Assumes artifacts already exist under:
      gs://<bucket>/runs/<run_id>/<label_tag>/
      - model_<label>.joblib
      - isotonic_calibrator_<label>.joblib
      - best_params.json
      - threshold_expected_<label>.txt
      - plots and CSVs (optional)

    Returns:
      {"artifact_uri": <prefix>, "model_resource_name": <vertex model name>}
    """
    cfg = load_env_config()
    project_id = cfg.get("project_id")
    region = cfg.get("region")

    # Initialize Vertex AI SDK
    init_ai(project_id, region)

    gcs_prefix = f"{gcs_model_bucket.rstrip('/')}/runs/{run_id}/{label_tag}"

    files = {
        "model_joblib": _gcs_join(gcs_prefix, f"model_{label_tag}.joblib"),
        "calibrator": _gcs_join(gcs_prefix, f"isotonic_calibrator_{label_tag}.joblib"),
        "best_params": _gcs_join(gcs_prefix, "best_params.json"),
        "threshold": _gcs_join(gcs_prefix, f"threshold_expected_{label_tag}.txt"),
        "feature_names": _gcs_join(gcs_prefix, "feature_names.json"),
    }

    manifest: Dict[str, object] = {
        "run_id": run_id,
        "label": label_tag,
        "files": files,
        "display_name": vertex_model_display_name,
        "registry_group": vertex_model_registry_label,
    }
    if best_params is not None:
        manifest["best_params"] = best_params
    if threshold is not None:
        manifest["threshold_expected"] = float(threshold)

    manifest_uri = _gcs_join(gcs_prefix, "manifest.json")
    gcs_upload_json(manifest_uri, manifest, indent=2)

    run_id_label = sanitize_for_bq_label(run_id)
    labels = {
        "stage": "candidate",
        "run_id": run_id_label,
        "label": label_tag,
        "group": vertex_model_registry_label,
    }
    metadata = {}
    if "threshold_expected" in manifest:
        metadata["threshold_expected"] = str(manifest["threshold_expected"])
    if "best_params" in manifest:
        metadata["best_params_json"] = json.dumps(manifest["best_params"], separators=(",", ":"))

    # Register with both a generic alias, a run-specific alias, and a label-specific production alias.
    run_id_label = sanitize_for_bq_label(run_id)
    run_label_alias = sanitize_for_bq_label(f"{run_id}_{label_tag}")
    model = register_model_version(
        display_name=vertex_model_display_name,
        artifact_uri=gcs_prefix,
        labels=labels,
        metadata=metadata,
        version_aliases=[
            "candidate",
            run_label_alias,
            f"production-{label_tag}",
        ],  # add unique + label-specific production aliases
    )


    return {
        "artifact_uri": gcs_prefix,
        "model_resource_name": model.resource_name,
    }

