-- Creates the daily inference feature table if it does not yet exist.
-- Dataset defaults to propensity_to_subscribe in project economedia-data-prod-laoy.

CREATE SCHEMA IF NOT EXISTS `economedia-data-prod-laoy.propensity_to_subscribe`;

CREATE TABLE IF NOT EXISTS `economedia-data-prod-laoy.propensity_to_subscribe.features_daily`
(
  scoring_date DATE   NOT NULL,  -- partition column
  user_id      INT64  NOT NULL,
  date         DATE   NOT NULL,

  active_cap_count INT64,
  active_dne_count INT64,

  pv_cap_7d   INT64,
  pv_cap_30d  INT64,
  pv_cap_60d  INT64,
  pv_cap_90d  INT64,
  pv_cap_all  INT64,
  pv_dne_7d   INT64,
  pv_dne_30d  INT64,
  pv_dne_60d  INT64,
  pv_dne_90d  INT64,
  pv_dne_all  INT64,

  walls_offer_cap_7d   INT64,
  walls_offer_cap_30d  INT64,
  walls_offer_cap_60d  INT64,
  walls_offer_cap_90d  INT64,
  walls_offer_cap_all  INT64,
  walls_offer_dne_7d   INT64,
  walls_offer_dne_30d  INT64,
  walls_offer_dne_60d  INT64,
  walls_offer_dne_90d  INT64,
  walls_offer_dne_all  INT64,

  nl_cap_7d   INT64,
  nl_cap_30d  INT64,
  nl_cap_60d  INT64,
  nl_cap_90d  INT64,
  nl_cap_all  INT64,
  nl_dne_7d   INT64,
  nl_dne_30d  INT64,
  nl_dne_60d  INT64,
  nl_dne_90d  INT64,
  nl_dne_all  INT64,

  subs_ended_cap_7d    INT64,
  subs_ended_cap_30d   INT64,
  subs_ended_cap_60d   INT64,
  subs_ended_cap_90d   INT64,
  subs_ended_cap_all   INT64,
  subs_ended_dne_7d    INT64,
  subs_ended_dne_30d   INT64,
  subs_ended_dne_60d   INT64,
  subs_ended_dne_90d   INT64,
  subs_ended_dne_all   INT64,
  subs_ended_other_7d  INT64,
  subs_ended_other_30d INT64,
  subs_ended_other_60d INT64,
  subs_ended_other_90d INT64,
  subs_ended_other_all INT64,

  pv_cap_ratio_7_30   FLOAT64,
  pv_cap_ratio_30_90  FLOAT64,
  pv_dne_ratio_7_30   FLOAT64,
  pv_dne_ratio_30_90  FLOAT64,
  walls_cap_ratio_7_30  FLOAT64,
  walls_cap_ratio_30_90 FLOAT64,
  walls_dne_ratio_7_30  FLOAT64,
  walls_dne_ratio_30_90 FLOAT64,
  nl_cap_ratio_7_30     FLOAT64,
  nl_cap_ratio_30_90    FLOAT64,
  nl_dne_ratio_7_30     FLOAT64,
  nl_dne_ratio_30_90    FLOAT64,
  subs_cap_ratio_7_30   FLOAT64,
  subs_cap_ratio_30_90  FLOAT64,
  subs_dne_ratio_7_30   FLOAT64,
  subs_dne_ratio_30_90  FLOAT64,

  recency_cap    INT64,
  frequency_cap  INT64,
  value_cap      FLOAT64,
  rfv_cap        FLOAT64,
  recency_dne    INT64,
  frequency_dne  INT64,
  value_dne      FLOAT64,
  rfv_dne        FLOAT64,

  total_paid_so_far                    FLOAT64,
  total_unsuccessful_payments_so_far   INT64,

  had_prev_free_cap  INT64,
  had_prev_free_dne  INT64,

  total_days_prev_cap    INT64,
  total_days_prev_dne    INT64,
  total_days_prev_other  INT64,

  last_payment_outcome INT64,

  active_cap_valid BOOL,
  active_dne_valid BOOL,

  education STRING,
  workpos   STRING,
  sex       STRING,

  miss_recency_cap     INT64,
  miss_recency_dne     INT64,
  miss_frequency_cap   INT64,
  miss_frequency_dne   INT64,
  miss_value_cap       INT64,
  miss_value_dne       INT64,
  miss_rfv_cap         INT64,
  miss_rfv_dne         INT64
)
PARTITION BY scoring_date
CLUSTER BY user_id;
