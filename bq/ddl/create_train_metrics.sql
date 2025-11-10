-- BigQuery DDL for training metrics tables.
-- Assumes dataset `propensity_to_subscribe` already exists.

CREATE SCHEMA IF NOT EXISTS `economedia-data-prod-laoy.propensity_to_subscribe`;

CREATE TABLE IF NOT EXISTS `economedia-data-prod-laoy.propensity_to_subscribe.train_metrics`
(
  run_id             STRING    NOT NULL,
  label_tag          STRING    NOT NULL,
  auc_val_mean       FLOAT64   NOT NULL,
  threshold_expected FLOAT64   NOT NULL,
  params_json        STRING    NOT NULL,
  created_at         TIMESTAMP NOT NULL
)
PARTITION BY DATE(created_at)
CLUSTER BY label_tag;

CREATE TABLE IF NOT EXISTS `economedia-data-prod-laoy.propensity_to_subscribe.train_metrics_detail`
(
  run_id                     STRING    NOT NULL,
  label                      STRING    NOT NULL,
  split                      STRING    NOT NULL,
  fold                       INT64     NOT NULL,
  n                          INT64,
  pos                        INT64,
  neg                        INT64,
  auc                        FLOAT64,
  pr_auc                     FLOAT64,
  logloss                    FLOAT64,
  brier                      FLOAT64,
  ks                         FLOAT64,
  precision                  FLOAT64,
  recall                     FLOAT64,
  accuracy                   FLOAT64,
  f1                         FLOAT64,
  threshold                  FLOAT64,
  tp                         FLOAT64,
  fp                         FLOAT64,
  tn                         FLOAT64,
  fn                         FLOAT64,
  exp_pos                    INT64,
  exp_neg                    INT64,
  exp_pr_auc                 FLOAT64,
  exp_logloss                FLOAT64,
  exp_brier                  FLOAT64,
  exp_tp                     FLOAT64,
  exp_fp                     FLOAT64,
  exp_tn                     FLOAT64,
  exp_fn                     FLOAT64,
  exp_precision              FLOAT64,
  exp_recall                 FLOAT64,
  exp_accuracy               FLOAT64,
  exp_f1                     FLOAT64,
  tpr                        FLOAT64,
  fpr                        FLOAT64,
  base_prob_obs_auc          FLOAT64,
  base_prob_obs_pr_auc       FLOAT64,
  base_prob_obs_logloss      FLOAT64,
  base_prob_obs_brier        FLOAT64,
  base_prob_exp_auc          FLOAT64,
  base_prob_exp_pr_auc       FLOAT64,
  base_prob_exp_logloss      FLOAT64,
  base_prob_exp_brier        FLOAT64,
  base_rand_obs_tp           FLOAT64,
  base_rand_obs_fp           FLOAT64,
  base_rand_obs_tn           FLOAT64,
  base_rand_obs_fn           FLOAT64,
  base_rand_obs_precision    FLOAT64,
  base_rand_obs_recall       FLOAT64,
  base_rand_obs_accuracy     FLOAT64,
  base_rand_obs_f1           FLOAT64,
  base_rand_exp_tp           FLOAT64,
  base_rand_exp_fp           FLOAT64,
  base_rand_exp_tn           FLOAT64,
  base_rand_exp_fn           FLOAT64,
  base_rand_exp_precision    FLOAT64,
  base_rand_exp_recall       FLOAT64,
  base_rand_exp_accuracy     FLOAT64,
  base_rand_exp_f1           FLOAT64,
  base_rate_pred             FLOAT64,
  base_p_const               FLOAT64,
  ingested_at                TIMESTAMP NOT NULL
)
CLUSTER BY label, split;
