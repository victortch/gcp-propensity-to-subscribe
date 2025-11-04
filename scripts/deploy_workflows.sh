#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ID:-economedia-data-prod-laoy}"
REGION="${REGION:-europe-west3}"
TRAINING_WORKFLOW="training-workflow"
INFERENCE_WORKFLOW="inference-workflow"

log() {
  printf "[%s] %s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

log "Deploying training workflow..."
gcloud workflows deploy "$TRAINING_WORKFLOW" \
  --source=workflows/training_workflow.yaml \
  --location="$REGION" \
  --project="$PROJECT"

log "Deploying inference workflow..."
gcloud workflows deploy "$INFERENCE_WORKFLOW" \
  --source=workflows/inference_workflow.yaml \
  --location="$REGION" \
  --project="$PROJECT"

log "Set service accounts separately using gcloud workflows update --service-account."
