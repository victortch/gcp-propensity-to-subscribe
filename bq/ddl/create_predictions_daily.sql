-- Creates predictions table if it does not exist.
-- Dataset defaults to propensity_to_subscribe in project economedia-data-prod-laoy.

CREATE SCHEMA IF NOT EXISTS `economedia-data-prod-laoy.propensity_to_subscribe`;

CREATE TABLE IF NOT EXISTS `economedia-data-prod-laoy.propensity_to_subscribe.predictions_daily`
(
  scoring_date  DATE     NOT NULL,  -- partition column
  user_id       INT64    NOT NULL,
  label         STRING   NOT NULL,  -- e.g. cap_30d
  prob          FLOAT64  NOT NULL,  -- calibrated probability
  decision      INT64    NOT NULL,  -- 0/1 via selected threshold
  threshold     FLOAT64,            -- threshold applied for decision
  model_version STRING,             -- Vertex AI model resource name
  artifact_uri  STRING,             -- GCS folder with artifacts
  model_run_id  STRING,             -- training run_id for the artifacts
  created_at    TIMESTAMP           -- ingestion timestamp (UTC)
)
PARTITION BY scoring_date
CLUSTER BY label, user_id;
