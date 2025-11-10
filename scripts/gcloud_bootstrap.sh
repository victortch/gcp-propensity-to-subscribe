#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ID:-economedia-data-prod-laoy}"
REGION="${REGION:-europe-west3}"
DATASET="${BQ_DATASET:-propensity_to_subscribe}"
REPO="${ARTIFACT_REPO:-pts_model}"
BUCKET="${GCS_BUCKET:-economedia-pts-models}"

log() {
  printf "[%s] %s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

log "Enabling core APIs..."
APIS=(
  aiplatform.googleapis.com
  bigquery.googleapis.com
  bigquerystorage.googleapis.com
  bigquerydatatransfer.googleapis.com
  artifactregistry.googleapis.com
  cloudbuild.googleapis.com
  storage.googleapis.com
  workflows.googleapis.com
  cloudscheduler.googleapis.com
  logging.googleapis.com
  monitoring.googleapis.com
  compute.googleapis.com
)
for api in "${APIS[@]}"; do
  log "Enabling API: $api"
  gcloud services enable "$api" --project="$PROJECT"
done

log "Creating Artifact Registry repo if missing..."
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --project="$PROJECT" \
  --description="Economedia PTS containers" \
  --quiet || log "Repository $REPO already exists."

log "Creating GCS bucket if missing..."
gsutil mb -p "$PROJECT" -l "$REGION" "gs://$BUCKET" || log "Bucket gs://$BUCKET already exists."

gsutil bucketpolicyonly set on "gs://$BUCKET" >/dev/null 2>&1 || true

gsutil label ch -l purpose:pts-models "gs://$BUCKET" >/dev/null 2>&1 || true

log "Ensuring BigQuery dataset $DATASET exists..."
if ! bq --project_id="$PROJECT" ls "$DATASET" >/dev/null 2>&1; then
  bq --project_id="$PROJECT" --location="$REGION" mk "$DATASET"
fi

log "Bootstrap complete."
