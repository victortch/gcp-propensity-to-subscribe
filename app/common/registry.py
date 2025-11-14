"""Vertex AI Model Registry helpers for the Economedia PTS project.

Notes
-----
- Uses the high-level :mod:`google.cloud.aiplatform` SDK.
- Artifacts (models, calibrators, params) remain in GCS; the Registry is a
  discovery/governance layer pointing at those artifacts via ``artifact_uri``.
- Version aliases (e.g., ``"candidate"``, ``"production"``) are used to promote/demote.

Authentication
--------------
- Locally: ``gcloud auth application-default login``
- In Vertex AI: service account attached to the Custom Job.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from google.cloud import aiplatform


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

def init_ai(project_id: str, region: str) -> None:
    """Initialize the Vertex AI SDK (must be called before using other funcs)."""
    aiplatform.init(project=project_id, location=region)


# ---------------------------------------------------------------------
# Registration / metadata
# ---------------------------------------------------------------------

def register_model_version(
    *,
    display_name: str,
    artifact_uri: str,
    labels: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, str]] = None,
    version_aliases: Optional[Iterable[str]] = None,
    serving_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
) -> aiplatform.Model:
    """
    Register a model version in Vertex AI Model Registry.

    Args:
        display_name: Human-friendly model name (stable across versions), e.g. "pts_xgb_model".
        artifact_uri: GCS folder containing the true artifacts (GCS is the source of truth).
        labels: Optional labels (e.g., {"stage": "candidate", "run_id": "..."}).
        metadata: Optional metadata (e.g., build window, git SHA, SQL SHA).
        version_aliases: Optional aliases, e.g., ["candidate"] or ["production"].
        serving_image_uri: We supply a generic prediction image to satisfy Registry fields,
                          even if we don't serve online from this Model.

    Returns:
        aiplatform.Model (the newly registered version)
    """
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_image_uri,
        labels=labels or {},
        metadata=metadata or {},
        version_aliases=list(version_aliases or []),
        # No need to set explanation or predict schemata for batch-only workflows.
    )
    return model


# ---------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------

def list_models_by_display_name(display_name: str) -> List[aiplatform.Model]:
    """List all versions under a given display name."""
    return list(aiplatform.Model.list(filter=f'display_name="{display_name}"'))


def list_models_by_label(label_key: str, label_value: str) -> List[aiplatform.Model]:
    """List models filtered by a single label key=value."""
    return list(aiplatform.Model.list(filter=f'labels.{label_key}="{label_value}"'))


def resolve_production_version(display_name: str) -> Optional[aiplatform.Model]:
    """
    Return the model version tagged with alias 'production', or None if not found.
    """
    for m in aiplatform.Model.list(filter=f'display_name="{display_name}"'):
        # version_aliases is a list of strings; check case-insensitively for robustness
        aliases = set(a.lower() for a in (m.version_aliases or []))
        if "production" in aliases:
            return m
    return None


def resolve_latest_version(display_name: str) -> Optional[aiplatform.Model]:
    """
    Return the most recently created version under a display name (best effort).
    """
    models = list(aiplatform.Model.list(filter=f'display_name="{display_name}"'))
    if not models:
        return None
    # aiplatform.Model has .create_time (RFC3339). Sort descending.
    models.sort(key=lambda m: m.gca_resource.create_time, reverse=True)
    return models[0]


def resolve_production_version_for_label(display_name: str, label_tag: str) -> Optional[aiplatform.Model]:
    """
    Return the model version for a specific label, preferring alias
    'production-{label_tag}', then global 'production'. If neither exists,
    return the most-recent version for that label. Returns None if no
    versions exist for this label.
    """
    models = list(aiplatform.Model.list(filter=f'display_name="{display_name}"'))
    if not models:
        return None

    def _labels(model: aiplatform.Model) -> Dict[str, str]:
        return dict(getattr(model, "labels", {}) or {})

    candidates = [m for m in models if _labels(m).get("label") == label_tag]
    if not candidates:
        return None

    wanted_alias = f"production-{label_tag}".lower()
    for model in candidates:
        aliases = set(alias.lower() for alias in (model.version_aliases or []))
        if wanted_alias in aliases:
            return model

    for model in candidates:
        aliases = set(alias.lower() for alias in (model.version_aliases or []))
        if "production" in aliases:
            return model

    candidates.sort(key=lambda m: m.gca_resource.create_time, reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------
# Aliases (promotion/demotion)
# ---------------------------------------------------------------------

def add_version_aliases(model: aiplatform.Model, aliases: Iterable[str]) -> None:
    """Add one or more aliases to this version (idempotent)."""
    model.add_version_aliases(list(aliases))


def remove_version_aliases(model: aiplatform.Model, aliases: Iterable[str]) -> None:
    """Remove one or more aliases from this version (idempotent)."""
    model.remove_version_aliases(list(aliases))


def promote_to_production(
    *,
    display_name: str,
    candidate_model: Optional[aiplatform.Model] = None,
    demote_existing: bool = True,
) -> aiplatform.Model:
    """
    Promote a model version to 'production' by setting version alias.

    If demote_existing=True, removes 'production' alias from any other versions.
    Returns the promoted model.
    """
    if candidate_model is None:
        candidate_model = resolve_latest_version(display_name)
        if candidate_model is None:
            raise ValueError(f"No versions found for display_name={display_name}")

    if demote_existing:
        for m in aiplatform.Model.list(filter=f'display_name="{display_name}"'):
            if m.resource_name != candidate_model.resource_name and m.version_aliases:
                if any(a.lower() == "production" for a in m.version_aliases):
                    remove_version_aliases(m, ["production"])

    add_version_aliases(candidate_model, ["production"])
    return candidate_model


# ---------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------

def get_artifact_uri(model: aiplatform.Model) -> Optional[str]:
    """Return the artifact URI (GCS path) recorded with the model version."""
    # aiplatform.Model exposes underlying gca_resource fields
    return getattr(model, "artifact_uri", None)


def get_labels(model: aiplatform.Model) -> Dict[str, str]:
    return dict(getattr(model, "labels", {}) or {})


def get_metadata(model: aiplatform.Model) -> Dict[str, str]:
    return dict(getattr(model, "metadata", {}) or {})
