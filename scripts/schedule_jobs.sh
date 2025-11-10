#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ID:-economedia-data-prod-laoy}"
REGION="${REGION:-europe-west3}"
WORKFLOW_REGION="$REGION"
TRAINING_WORKFLOW="training-workflow"
INFERENCE_WORKFLOW="inference-workflow"

TRAINING_SCHEDULE="0 6 1 */3 *"   # 06:00 UTC on the first day of every 3rd month
INFERENCE_SCHEDULE="0 3 * * *"    # 03:00 UTC daily

# Build scheduler payloads with optional overrides supplied via environment
# variables. If no override is provided, Cloud Workflows derives sensible
# defaults (e.g., freeze_date = today-7d, scoring_date = yesterday).
TRAINING_BODY=$(PROJECT="$PROJECT" REGION="$REGION" python3 - <<'PY'
import json
import os

payload = {
    "project_id": os.environ["PROJECT"],
    "region": os.environ["REGION"],
}

overrides = [
    ("TRAINING_START_DATE", "start_date"),
    ("TRAINING_FREEZE_DATE", "freeze_date"),
    ("TRAINING_DNE_START", "dne_start"),
    ("TRAINING_CAP_WALL1_OFFER_START", "cap_wall1_offer_start"),
    ("TRAINING_PRIMARY_LABEL", "primary_label"),
    ("TRAINING_IMAGE", "training_image"),
    ("TRAINING_MACHINE_TYPE", "machine_type"),
    ("TRAINING_SERVICE_ACCOUNT", "service_account"),
]

for env_key, arg_key in overrides:
    value = os.environ.get(env_key)
    if value:
        payload[arg_key] = value

print(json.dumps({"argument": json.dumps(payload)}))
PY
)

INFERENCE_BODY=$(PROJECT="$PROJECT" REGION="$REGION" python3 - <<'PY'
import json
import os

payload = {
    "project_id": os.environ["PROJECT"],
    "region": os.environ["REGION"],
}

overrides = [
    ("INFERENCE_SCORING_DATE", "scoring_date"),
    ("INFERENCE_IMAGE", "inference_image"),
    ("INFERENCE_MACHINE_TYPE", "machine_type"),
    ("INFERENCE_SERVICE_ACCOUNT", "service_account"),
]

for env_key, arg_key in overrides:
    value = os.environ.get(env_key)
    if value:
        payload[arg_key] = value

print(json.dumps({"argument": json.dumps(payload)}))
PY
)

log() {
  printf "[%s] %s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

log "Configuring training scheduler..."
if ! gcloud scheduler jobs describe training-quarterly --project="$PROJECT" >/dev/null 2>&1; then
  gcloud scheduler jobs create http training-quarterly \
    --project="$PROJECT" \
    --schedule="$TRAINING_SCHEDULE" \
    --uri="https://workflowexecutions.googleapis.com/v1/projects/$PROJECT/locations/$WORKFLOW_REGION/workflows/$TRAINING_WORKFLOW/executions" \
    --http-method=POST \
    --time-zone="Etc/UTC" \
    --oidc-service-account-email="sa-workflows@${PROJECT}.iam.gserviceaccount.com" \
    --headers="Content-Type=application/json" \
    --message-body="$TRAINING_BODY"
else
  gcloud scheduler jobs update http training-quarterly \
    --project="$PROJECT" \
    --schedule="$TRAINING_SCHEDULE" \
    --uri="https://workflowexecutions.googleapis.com/v1/projects/$PROJECT/locations/$WORKFLOW_REGION/workflows/$TRAINING_WORKFLOW/executions" \
    --http-method=POST \
    --time-zone="Etc/UTC" \
    --oidc-service-account-email="sa-workflows@${PROJECT}.iam.gserviceaccount.com" \
    --headers="Content-Type=application/json" \
    --message-body="$TRAINING_BODY"
fi

log "Configuring inference scheduler..."
if ! gcloud scheduler jobs describe inference-daily --project="$PROJECT" >/dev/null 2>&1; then
  gcloud scheduler jobs create http inference-daily \
    --project="$PROJECT" \
    --schedule="$INFERENCE_SCHEDULE" \
    --uri="https://workflowexecutions.googleapis.com/v1/projects/$PROJECT/locations/$WORKFLOW_REGION/workflows/$INFERENCE_WORKFLOW/executions" \
    --http-method=POST \
    --time-zone="Etc/UTC" \
    --oidc-service-account-email="sa-workflows@${PROJECT}.iam.gserviceaccount.com" \
    --headers="Content-Type=application/json" \
    --message-body="$INFERENCE_BODY"
else
  gcloud scheduler jobs update http inference-daily \
    --project="$PROJECT" \
    --schedule="$INFERENCE_SCHEDULE" \
    --uri="https://workflowexecutions.googleapis.com/v1/projects/$PROJECT/locations/$WORKFLOW_REGION/workflows/$INFERENCE_WORKFLOW/executions" \
    --http-method=POST \
    --time-zone="Etc/UTC" \
    --oidc-service-account-email="sa-workflows@${PROJECT}.iam.gserviceaccount.com" \
    --headers="Content-Type=application/json" \
    --message-body="$INFERENCE_BODY"
fi

log "Scheduler configuration complete."
