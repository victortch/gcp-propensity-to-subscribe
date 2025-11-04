#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ID:-economedia-data-prod-laoy}"
REGION="${REGION:-europe-west3}"
DATASET="${BQ_DATASET:-propensity_to_subscribe}"
DESTINATION_TABLE="${DEST_TABLE:-propensity_to_subscribe.features_daily}"
CONFIG_NAME="features-daily"
SCHEDULE="every 24 hours"
TIME_ZONE="Europe/Sofia"

log() {
  printf "[%s] %s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

SQL=$(python - <<'PY'
from pathlib import Path
import json
sql = Path('bq/scheduled_queries/features_daily.sql').read_text()
print(json.dumps(sql)[1:-1])
PY
)

TMP_JSON=$(mktemp)
cat >"$TMP_JSON" <<JSON
{
  "name": "$CONFIG_NAME",
  "displayName": "Economedia PTS features daily",
  "dataSourceId": "scheduled_query",
  "schedule": "$SCHEDULE",
  "params": {
    "query": "$SQL",
    "destination_table_name_template": "$DESTINATION_TABLE",
    "write_disposition": "WRITE_APPEND",
    "partitioning_field": "scoring_date"
  },
  "destinationDatasetId": "$DATASET",
  "location": "$REGION"
}
JSON

log "Creating or updating scheduled query transfer config..."
if bq --project_id="$PROJECT" --location="$REGION" ls --transfer_config | grep -q "$CONFIG_NAME"; then
  bq --project_id="$PROJECT" --location="$REGION" update --transfer_config "$CONFIG_NAME" "$TMP_JSON"
else
  bq --project_id="$PROJECT" --location="$REGION" mk --transfer_config "$TMP_JSON"
fi

rm -f "$TMP_JSON"
log "Scheduled query configuration ready."
