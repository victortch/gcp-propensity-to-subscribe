#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ID:-economedia-data-prod-laoy}"
REGION="${REGION:-europe-west3}"
WORKFLOW_REGION="$REGION"
TRAINING_WORKFLOW="training-workflow"
INFERENCE_WORKFLOW="inference-workflow"

TRAINING_SCHEDULE="0 6 1 */3 *"   # 06:00 UTC on the first day of every 3rd month
INFERENCE_SCHEDULE="0 3 * * *"    # 03:00 UTC daily

TRAINING_BODY=$(printf '{"argument":"{\\"project_id\\":\\"%s\\",\\"region\\":\\"%s\\"}"}' "$PROJECT" "$REGION")
INFERENCE_BODY=$(printf '{"argument":"{\\"project_id\\":\\"%s\\",\\"region\\":\\"%s\\",\\"scoring_date\\":\\"%s\\"}"}' "$PROJECT" "$REGION" "$(date -u +%Y-%m-%d)")

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
