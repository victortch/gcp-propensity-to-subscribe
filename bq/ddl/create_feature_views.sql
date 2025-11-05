-- Convenience views for analysts and monitoring.

CREATE OR REPLACE VIEW `economedia-data-prod-laoy.propensity_to_subscribe.latest_train_runs` AS
SELECT AS STRUCT
  run_id,
  label_tag,
  auc_val_mean,
  threshold_expected,
  params_json,
  created_at,
  ROW_NUMBER() OVER (PARTITION BY label_tag ORDER BY created_at DESC) AS recency_rank
FROM `economedia-data-prod-laoy.propensity_to_subscribe.train_metrics`;

CREATE OR REPLACE VIEW `economedia-data-prod-laoy.propensity_to_subscribe.predictions_latest` AS
SELECT
  * EXCEPT(row_num)
FROM (
  SELECT
    p.*, 
    ROW_NUMBER() OVER (PARTITION BY user_id, label ORDER BY scoring_date DESC, created_at DESC) AS row_num
  FROM `economedia-data-prod-laoy.propensity_to_subscribe.predictions_daily` p
)
WHERE row_num = 1;
