-- Daily feature materialization for inference.
-- Runs as a BigQuery scheduled query.
-- Ensures features match the training dataset logic for the requested date.

DECLARE run_date DATE DEFAULT @run_date;
DECLARE target_date DATE DEFAULT IFNULL(run_date, DATE_SUB(CURRENT_DATE('Europe/Sofia'), INTERVAL 1 DAY));

-- Match training's default horizon: start_date = freeze_date - 180d.
-- For inference we set start_date = target_date - 180d so that "*_all" windows
-- have the same scale the model was trained on near the freeze date.
DECLARE window_days INT64 DEFAULT 180;
DECLARE start_date DATE DEFAULT DATE_SUB(target_date, INTERVAL window_days DAY);

-- Use the same default semantics as training workflow (defaults to start_date).
DECLARE dne_start DATE DEFAULT start_date;
DECLARE cap_wall1_offer_start DATE DEFAULT start_date;


-- Remove any existing rows for the target date before inserting fresh features.
DELETE FROM `economedia-data-prod-laoy.propensity_to_subscribe.features_daily`
WHERE scoring_date = target_date;

-- Insert rebuilt feature rows.
INSERT INTO `economedia-data-prod-laoy.propensity_to_subscribe.features_daily`
-- TRAINING DATASET ASSEMBLY (Capital & Dnevnik) --------------------------------
-- Reads once per run; returns user-date rows with feature inputs only (no labels)

WITH
-- 0) Corporate users to exclude
corporate AS (
  SELECT DISTINCT lguserid FROM `economedia-data-prod-laoy.public.lgusers_corporate` WHERE lguserid IS NOT NULL
  UNION DISTINCT
  SELECT DISTINCT ownerid AS lguserid FROM `economedia-data-prod-laoy.public.lgusers_corporate` WHERE ownerid IS NOT NULL
),

-- 1) Manual stops
stop_log AS (
  SELECT orderid, MIN(datetime) AS stop_date
  FROM `economedia-data-prod-laoy.public.payments_recurring_stop_log`
  GROUP BY 1
),

-- 2) Raw orders joined (minimal columns)
orders_raw AS (
  SELECT
    a.lguserid,
    a.id         AS order_id,
    b.id         AS orderdetail_id,
    c.orderdetailsid AS ordersubscription_id,
    a.createdate AS created_ts,              -- TIMESTAMP UTC
    a.state,                                 -- 1 = success
    a.totalprice,
    b.category,
    c.izdanie,
    c.period,
    c.perioddays,
    CASE
      WHEN b.category IN (1,2) AND c.izdanie IN (1,2)   THEN 'capital'
      WHEN b.category = 10    AND c.izdanie = 200       THEN 'dnevnik'
      ELSE 'other'
    END AS media,
    sl.stop_date
  FROM `economedia-data-prod-laoy.ecommerce.orders` a
  JOIN `economedia-data-prod-laoy.ecommerce.orderdetails` b ON a.id = b.orderid
  LEFT JOIN `economedia-data-prod-laoy.ecommerce.ordersubscription` c ON b.id = c.orderdetailsid
  LEFT JOIN stop_log sl ON a.id = sl.orderid
  WHERE DATE(a.createdate) BETWEEN start_date AND target_date
    AND a.lguserid IS NOT NULL
    AND a.lguserid NOT IN (SELECT lguserid FROM corporate)
),

-- 3) Successful orders → "subscription starts" (valid & invalid)
subs_orders AS (
  SELECT
    lguserid, media, order_id, created_ts,
    SAFE_CAST(period AS INT64) AS period_mo,
    perioddays,
    totalprice,
    state,
    -- Validity for labels & guard:
    (state = 1 AND totalprice <> 0 AND SAFE_CAST(period AS INT64) IN (1,6,12) AND perioddays = 30) AS is_valid,
    -- Due date primarily by months (matches your example)
    TIMESTAMP_ADD(created_ts, INTERVAL COALESCE(SAFE_CAST(period AS INT64), 0)*30 DAY) AS due_ts,
    stop_date
  FROM orders_raw
  WHERE state = 1
),

-- 4) Next successful order per (user, media) to bridge coverage
subs_orders_sorted AS (
  SELECT
    s.*,
    LEAD(created_ts) OVER (PARTITION BY lguserid, media ORDER BY created_ts, order_id) AS next_success_ts
  FROM subs_orders s
),

-- 5) Coverage intervals (start_ts .. end_ts)
subs_intervals AS (
  SELECT
    lguserid, media, is_valid,
    created_ts AS start_ts,
    LEAST(
      COALESCE(stop_date, TIMESTAMP '9999-12-31 00:00:00 UTC'),
      COALESCE(next_success_ts, TIMESTAMP '9999-12-31 00:00:00 UTC'),
      TIMESTAMP_ADD(due_ts, INTERVAL 90 DAY)
    ) AS end_ts
  FROM subs_orders_sorted
),

subs_intervals_dated AS (
  SELECT
    lguserid, media, is_valid,
    DATE(start_ts) AS start_date,
    DATE(end_ts)   AS end_date,
    GREATEST(DATE_DIFF(DATE(end_ts), DATE(start_ts), DAY), 0) AS days_covered
  FROM subs_intervals
),

-- 6) "Sub ended" daily events (for previous-sub counts; by media+ "other")
subs_end_events AS (
  SELECT lguserid, media, end_date AS date, COUNT(*) AS subs_ended_cnt
  FROM subs_intervals_dated
  GROUP BY 1,2,3
),

-- 7) Accumulate "total days previously subscribed" by media(cumulated by end_date)
subs_days_by_end_date AS (
  SELECT lguserid, media, end_date AS date, SUM(days_covered) AS days_covered_sum
  FROM subs_intervals_dated
  GROUP BY 1,2,3
),
subs_days_cum AS (
  SELECT
    lguserid, media, date,
    SUM(days_covered_sum)
      OVER (PARTITION BY lguserid, media ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_days_covered_upto_date
  FROM subs_days_by_end_date
),

-- 8) Orders daily aggregates for payments & failures (all media)
orders_daily AS (
  SELECT
    lguserid,
    DATE(created_ts) AS date,
    COUNTIF(state=1) AS orders_success_cnt,
    COUNTIF(state<>1) AS orders_fail_cnt,
    SUM(CASE WHEN state=1 AND totalprice>0 THEN totalprice ELSE 0 END) AS paid_amount
  FROM orders_raw
  WHERE DATE(created_ts) BETWEEN start_date AND target_date
  GROUP BY 1,2
),
orders_cum AS (
  SELECT
    lguserid, date,
    SUM(orders_fail_cnt) OVER (PARTITION BY lguserid ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_fail_cnt,
    SUM(paid_amount) OVER (PARTITION BY lguserid ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_paid_amount
  FROM orders_daily
),

-- 9) "Ever free" (state=1 & totalprice=0) flags by media
free_sub_ever AS (
  SELECT lguserid, media, MIN(DATE(created_ts)) AS first_free_date
  FROM subs_orders_sorted
  WHERE totalprice = 0
  GROUP BY 1,2
),

-- 10) Pageviews per day (sid: 1=Capital, 2=Dnevnik); count every row
pv_daily AS (
  SELECT lguserid, DATE(datetime) AS date, sid, COUNT(*) AS pv_count
  FROM `economedia-data-prod-laoy.public.paywall_log_main`
  WHERE DATE(datetime) BETWEEN start_date AND target_date
    AND sid IN (1,2) AND COALESCE(lguserid,0) > 0
  GROUP BY 1,2,3
  UNION ALL
  SELECT lguserid, DATE(datetime) AS date, sid, COUNT(*) AS pv_count
  FROM `economedia-data-prod-laoy.public.paywall_log_archive_full`
  WHERE DATE(datetime) BETWEEN start_date AND target_date
    AND sid IN (1,2) AND COALESCE(lguserid,0) > 0
  GROUP BY 1,2,3
),
pv_daily_pivot AS (
  SELECT lguserid, date,
         SUM(CASE WHEN sid=1 THEN pv_count ELSE 0 END) AS pv_cap,
         SUM(CASE WHEN sid=2 THEN pv_count ELSE 0 END) AS pv_dne
  FROM pv_daily
  GROUP BY 1,2
),

-- Per-user "ever had PV" flags (Capital/Dnevnik)
pv_users AS (
  SELECT
    lguserid,
    -- any Capital PV at any time?
    SUM(CASE WHEN sid = 1 THEN pv_count ELSE 0 END) > 0 AS has_cap_pv,
    -- any Dnevnik PV at any time?
    SUM(CASE WHEN sid = 2 THEN pv_count ELSE 0 END) > 0 AS has_dne_pv
  FROM pv_daily
  GROUP BY lguserid
),

-- 11) Walls; build "with offer" flags per your rules
walls_daily AS (
  SELECT lguserid, DATE(date) AS date, sid, wall
  FROM `economedia-data-prod-laoy.public.lgusers_wall_full`
  WHERE DATE(date) BETWEEN start_date AND target_date
    AND lguserid IS NOT NULL
),
walls_offer_daily AS (
  SELECT
    lguserid, date,
    SUM(CASE
          WHEN sid=1 AND (wall IN (2,3,5,6) OR (wall=1 AND date >= cap_wall1_offer_start)) THEN 1 ELSE 0
        END) AS walls_offer_cap,
    SUM(CASE
          WHEN sid=2 AND wall IN (2,3) THEN 1 ELSE 0
        END) AS walls_offer_dne
  FROM walls_daily
  GROUP BY 1,2
),

-- 12) Newsletter events (dedupe per user/sid/day)
newsletter_daily AS (
  SELECT DISTINCT
    COALESCE(n.lguserid, lg.lguserid) AS lguserid,
    DATE(n.regdate) AS date,
    n.sid
  FROM `economedia-data-prod-laoy.public.newsletter` n
  LEFT JOIN `economedia-data-prod-laoy.public.lgusers` lg
    ON (lg.email = n.email OR n.lguserid = lg.lguserid)
  WHERE n.active = 1 AND n.sid IN (1,2) AND DATE(n.regdate) BETWEEN start_date AND target_date
),
newsletter_daily_pivot AS (
  SELECT lguserid, date,
         COUNTIF(sid=1) AS nl_cap,
         COUNTIF(sid=2) AS nl_dne
  FROM newsletter_daily
  GROUP BY 1,2
),

-- 13) RFV by day/sid (pick one per day if multiple)
rfv_raw AS (
  SELECT lguserid, DATE(date) AS date, sid, recency, frequency, volume, rfv
  FROM `economedia-data-prod-laoy.public.lgusers_rfv`
  WHERE sid IN (1,2) AND DATE(date) BETWEEN start_date AND target_date
),
rfv_daily AS (
  SELECT lguserid, date, sid,
         ANY_VALUE(recency)  AS recency,
         ANY_VALUE(frequency) AS frequency,
         ANY_VALUE(volume)    AS value,
         ANY_VALUE(rfv)      AS rfv
  FROM rfv_raw
  GROUP BY 1,2,3
),
rfv_pivot AS (
  SELECT lguserid, date,
    MAX(IF(sid=1, recency,  NULL)) AS recency_cap,
    MAX(IF(sid=1, frequency, NULL)) AS frequency_cap,
    MAX(IF(sid=1, value,    NULL)) AS value_cap,
    MAX(IF(sid=1, rfv,      NULL)) AS rfv_cap,
    MAX(IF(sid=2, recency,  NULL)) AS recency_dne,
    MAX(IF(sid=2, frequency, NULL)) AS frequency_dne,
    MAX(IF(sid=2, value,    NULL)) AS value_dne,
    MAX(IF(sid=2, rfv,      NULL)) AS rfv_dne
  FROM rfv_daily
  GROUP BY 1,2
),

-- 14) Spine of user–dates (has at least one record in any source except lgusersdet)
spine AS (
  SELECT lguserid, date FROM pv_daily_pivot
  UNION DISTINCT SELECT lguserid, date FROM walls_offer_daily
  UNION DISTINCT SELECT lguserid, date FROM newsletter_daily_pivot
  UNION DISTINCT SELECT lguserid, DATE(created_ts) FROM orders_raw
  UNION DISTINCT SELECT lguserid, date FROM rfv_pivot
),
index_candidates AS (
  SELECT lguserid, date
  FROM spine
  WHERE date BETWEEN DATE_ADD(start_date, INTERVAL 90 DAY) AND target_date

),

-- 15) Active flags on index dates (valid vs any)
index_active AS (
  SELECT
    idx.lguserid, idx.date,
    EXISTS (
      SELECT 1 FROM subs_intervals s
      WHERE s.lguserid = idx.lguserid AND s.media ='capital'
        AND TIMESTAMP(idx.date) >= s.start_ts AND TIMESTAMP(idx.date) < s.end_ts
    ) AS active_cap_any,
    EXISTS (
      SELECT 1 FROM subs_intervals s
      WHERE s.lguserid = idx.lguserid AND s.media ='capital' AND s.is_valid
        AND TIMESTAMP(idx.date) >= s.start_ts AND TIMESTAMP(idx.date) < s.end_ts
    ) AS active_cap_valid,
    EXISTS (
      SELECT 1 FROM subs_intervals s
      WHERE s.lguserid = idx.lguserid AND s.media ='dnevnik'
        AND TIMESTAMP(idx.date) >= s.start_ts AND TIMESTAMP(idx.date) < s.end_ts
    ) AS active_dne_any,
    EXISTS (
      SELECT 1 FROM subs_intervals s
      WHERE s.lguserid = idx.lguserid AND s.media ='dnevnik' AND s.is_valid
        AND TIMESTAMP(idx.date) >= s.start_ts AND TIMESTAMP(idx.date) < s.end_ts
    ) AS active_dne_valid
  FROM index_candidates idx
),

-- 16) Future subscription guard (get NEXT order after date; drop if NEXT is invalid)
-- 16) Future subscription guard (JOIN-based; pick earliest future order per brand)
next_flags_base AS (
  SELECT
    i.lguserid,
    i.date,
    s.media,
    s.is_valid,
    ROW_NUMBER() OVER (
      PARTITION BY i.lguserid, i.date, s.media
      ORDER BY s.created_ts
    ) AS rn
  FROM index_candidates i
  LEFT JOIN subs_orders_sorted s
    ON s.lguserid = i.lguserid
   AND s.created_ts > TIMESTAMP(i.date)
   AND s.media IN ('capital','dnevnik')
),
next_flags AS (
  SELECT
    lguserid, date,
    MAX(IF(media='capital' , is_valid, NULL)) AS next_cap_is_valid,
    MAX(IF(media='dnevnik', is_valid, NULL)) AS next_dne_is_valid
  FROM next_flags_base
  WHERE rn = 1
  GROUP BY lguserid, date
),



index_filtered AS (
  SELECT
    ia.lguserid,
    ia.date,
    ia.active_cap_any,
    ia.active_cap_valid,
    ia.active_dne_any,
    ia.active_dne_valid,
    nf.next_cap_is_valid,
    nf.next_dne_is_valid,
    (NOT ia.active_cap_valid AND NOT ia.active_cap_any AND (nf.next_cap_is_valid IS NULL OR nf.next_cap_is_valid = TRUE)) AS candidate_cap,
    (ia.date >= dne_start AND NOT ia.active_dne_valid AND NOT ia.active_dne_any AND (nf.next_dne_is_valid IS NULL OR nf.next_dne_is_valid = TRUE)) AS candidate_dne
  FROM index_active ia
  JOIN next_flags nf USING (lguserid, date)
  WHERE ia.date = target_date
    AND (
      (NOT ia.active_cap_valid AND NOT ia.active_cap_any AND (nf.next_cap_is_valid IS NULL OR nf.next_cap_is_valid = TRUE))
      OR
      (ia.date >= dne_start AND NOT ia.active_dne_valid AND NOT ia.active_dne_any AND (nf.next_dne_is_valid IS NULL OR nf.next_dne_is_valid = TRUE))
    )
),

-- 19) Build daily signal table (one row per user-date where any event happened)
daily_signal AS (
  SELECT
    d.lguserid, d.date,
    COALESCE(pv.pv_cap, 0)           AS pv_cap,
    COALESCE(pv.pv_dne, 0)           AS pv_dne,
    COALESCE(w.walls_offer_cap, 0)   AS walls_offer_cap,
    COALESCE(w.walls_offer_dne, 0)   AS walls_offer_dne,
    COALESCE(nl.nl_cap, 0)           AS nl_cap,
    COALESCE(nl.nl_dne, 0)           AS nl_dne,
    COALESCE(se_cap.subs_ended_cnt, 0)   AS subs_ended_cap,
    COALESCE(se_dne.subs_ended_cnt, 0)   AS subs_ended_dne,
    COALESCE(se_other.subs_ended_cnt, 0) AS subs_ended_other
  FROM (SELECT DISTINCT lguserid, date FROM spine) d
  LEFT JOIN pv_daily_pivot      pv ON pv.lguserid=d.lguserid AND pv.date=d.date
  LEFT JOIN walls_offer_daily    w ON w.lguserid=d.lguserid AND w.date=d.date
  LEFT JOIN newsletter_daily_pivot nl ON nl.lguserid=d.lguserid AND nl.date=d.date
  LEFT JOIN subs_end_events se_cap   ON se_cap.lguserid=d.lguserid AND se_cap.date=d.date AND se_cap.media ='capital'
  LEFT JOIN subs_end_events se_dne   ON se_dne.lguserid=d.lguserid AND se_dne.date=d.date AND se_dne.media ='dnevnik'
  LEFT JOIN subs_end_events se_other ON se_other.lguserid=d.lguserid AND se_other.date=d.date AND se_other.media ='other'
),

-- 20) Rolling windows (7/30/60/90/all) on the daily signal

-- 20) Rolling windows (7/30/60/90/all) on the daily signal  ✅ FIXED
-- Use a numeric order key because BigQuery's RANGE frames require numeric ORDER BY.
rolling_base AS (
  SELECT
    ds.*,
    UNIX_DATE(ds.date) AS od   -- numeric day index
  FROM daily_signal ds
),

rolling AS (
  SELECT
    rb.* EXCEPT(od),

    -- 7 days (inclusive: current day + prior 6)
    SUM(pv_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS pv_cap_7d,
    SUM(pv_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS pv_dne_7d,
    SUM(walls_offer_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS walls_offer_cap_7d,
    SUM(walls_offer_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS walls_offer_dne_7d,
    SUM(nl_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS nl_cap_7d,
    SUM(nl_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS nl_dne_7d,
    SUM(subs_ended_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS subs_ended_cap_7d,
    SUM(subs_ended_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS subs_ended_dne_7d,
    SUM(subs_ended_other) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS subs_ended_other_7d,

    -- 30 days
    SUM(pv_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 29 PRECEDING AND CURRENT ROW) AS pv_cap_30d,
    SUM(pv_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 29 PRECEDING AND CURRENT ROW) AS pv_dne_30d,
    SUM(walls_offer_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 29 PRECEDING AND CURRENT ROW) AS walls_offer_cap_30d,
    SUM(walls_offer_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 29 PRECEDING AND CURRENT ROW) AS walls_offer_dne_30d,
    SUM(nl_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 29 PRECEDING AND CURRENT ROW) AS nl_cap_30d,
    SUM(nl_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 29 PRECEDING AND CURRENT ROW) AS nl_dne_30d,
    SUM(subs_ended_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 29 PRECEDING AND CURRENT ROW) AS subs_ended_cap_30d,
    SUM(subs_ended_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 29 PRECEDING AND CURRENT ROW) AS subs_ended_dne_30d,
    SUM(subs_ended_other) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 29 PRECEDING AND CURRENT ROW) AS subs_ended_other_30d,

    -- 60 days
    SUM(pv_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 59 PRECEDING AND CURRENT ROW) AS pv_cap_60d,
    SUM(pv_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 59 PRECEDING AND CURRENT ROW) AS pv_dne_60d,
    SUM(walls_offer_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 59 PRECEDING AND CURRENT ROW) AS walls_offer_cap_60d,
    SUM(walls_offer_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 59 PRECEDING AND CURRENT ROW) AS walls_offer_dne_60d,
    SUM(nl_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 59 PRECEDING AND CURRENT ROW) AS nl_cap_60d,
    SUM(nl_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 59 PRECEDING AND CURRENT ROW) AS nl_dne_60d,
    SUM(subs_ended_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 59 PRECEDING AND CURRENT ROW) AS subs_ended_cap_60d,
    SUM(subs_ended_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 59 PRECEDING AND CURRENT ROW) AS subs_ended_dne_60d,
    SUM(subs_ended_other) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 59 PRECEDING AND CURRENT ROW) AS subs_ended_other_60d,

    -- 90 days
    SUM(pv_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 89 PRECEDING AND CURRENT ROW) AS pv_cap_90d,
    SUM(pv_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 89 PRECEDING AND CURRENT ROW) AS pv_dne_90d,
    SUM(walls_offer_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 89 PRECEDING AND CURRENT ROW) AS walls_offer_cap_90d,
    SUM(walls_offer_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 89 PRECEDING AND CURRENT ROW) AS walls_offer_dne_90d,
    SUM(nl_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 89 PRECEDING AND CURRENT ROW) AS nl_cap_90d,
    SUM(nl_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 89 PRECEDING AND CURRENT ROW) AS nl_dne_90d,
    SUM(subs_ended_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 89 PRECEDING AND CURRENT ROW) AS subs_ended_cap_90d,
    SUM(subs_ended_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 89 PRECEDING AND CURRENT ROW) AS subs_ended_dne_90d,
    SUM(subs_ended_other) OVER (PARTITION BY lguserid ORDER BY rb.od
      RANGE BETWEEN 89 PRECEDING AND CURRENT ROW) AS subs_ended_other_90d,

    -- All past (running totals) – row-based is fine; keep numeric ordering for stability
    SUM(pv_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS pv_cap_all,
    SUM(pv_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS pv_dne_all,
    SUM(walls_offer_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS walls_offer_cap_all,
    SUM(walls_offer_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS walls_offer_dne_all,
    SUM(nl_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS nl_cap_all,
    SUM(nl_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS nl_dne_all,
    SUM(subs_ended_cap) OVER (PARTITION BY lguserid ORDER BY rb.od
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS subs_ended_cap_all,
    SUM(subs_ended_dne) OVER (PARTITION BY lguserid ORDER BY rb.od
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS subs_ended_dne_all,
    SUM(subs_ended_other) OVER (PARTITION BY lguserid ORDER BY rb.od
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS subs_ended_other_all
  FROM rolling_base rb
),

-- Pick latest orders_cum row ≤ index date (per user-date)
oc_latest AS (
  SELECT
    idx.lguserid, idx.date,
    oc.cum_paid_amount, oc.cum_fail_cnt,
    ROW_NUMBER() OVER (
      PARTITION BY idx.lguserid, idx.date
      ORDER BY oc.date DESC
    ) AS rn
  FROM index_filtered idx
  LEFT JOIN orders_cum oc
    ON oc.lguserid = idx.lguserid
   AND oc.date     <= idx.date
),
oc_pick AS (
  SELECT
    lguserid,
    date, 
    IFNULL(cum_paid_amount, 0) AS total_paid_so_far,
    IFNULL(cum_fail_cnt, 0)    AS total_unsuccessful_payments_so_far
  FROM oc_latest
  WHERE rn = 1
),


-- Pick latest subs_days_cum row (< index date) per media
subs_days_latest AS (
  SELECT
    idx.lguserid, idx.date, s.media, s.cum_days_covered_upto_date,
    ROW_NUMBER() OVER (
      PARTITION BY idx.lguserid, idx.date, s.media
      ORDER BY s.date DESC
    ) AS rn
  FROM index_filtered idx
  LEFT JOIN subs_days_cum s
    ON s.lguserid = idx.lguserid
   AND s.date     <  idx.date
   AND s.media    IN ('capital','dnevnik','other')
),
subs_days_pick AS (
  SELECT
    lguserid, date,
    IFNULL(MAX(IF(media='capital', cum_days_covered_upto_date, NULL)), 0) AS total_days_prev_cap,
    IFNULL(MAX(IF(media='dnevnik', cum_days_covered_upto_date, NULL)), 0) AS total_days_prev_dne,
    IFNULL(MAX(IF(media='other',   cum_days_covered_upto_date, NULL)), 0) AS total_days_prev_other
  FROM subs_days_latest
  WHERE rn = 1
  GROUP BY lguserid, date
),

-- Last payment outcome before/on index date (0=no prior, 1=last unsuccessful, 2=last successful)
last_payment_latest AS (
  SELECT
    idx.lguserid, idx.date,
    CASE WHEN o.state = 1 THEN 2 ELSE 1 END AS last_payment_outcome,
    ROW_NUMBER() OVER (
      PARTITION BY idx.lguserid, idx.date
      ORDER BY o.created_ts DESC
    ) AS rn
  FROM index_filtered idx
  LEFT JOIN orders_raw o
    ON o.lguserid = idx.lguserid
   AND DATE(o.created_ts) <= idx.date
),
last_payment_pick AS (
  SELECT lguserid, date, IFNULL(last_payment_outcome, 0) AS last_payment_outcome
  FROM last_payment_latest
  WHERE rn = 1
),

-- Ever free subscription before index date (per media)
free_flags AS (
  SELECT
    idx.lguserid, idx.date,
    -- Exists any free sub earlier?
    IF(COUNTIF(f.media='capital')>0, 1, 0) AS had_prev_free_cap,
    IF(COUNTIF(f.media='dnevnik')>0, 1, 0) AS had_prev_free_dne
  FROM index_filtered idx
  LEFT JOIN free_sub_ever f
    ON f.lguserid = idx.lguserid
   AND f.first_free_date < idx.date
  GROUP BY idx.lguserid, idx.date
),


-- 21) Final assembly: feature view (joins rolling + rfv + orders cum + active flags + totals)
final AS (
  SELECT
    idx.lguserid AS user_id,
    idx.date,
    idx.candidate_cap,
    idx.candidate_dne,

    -- Current active counts per media (0/1)
    IF(idx.active_cap_valid, 1, 0) AS active_cap_count,
    IF(idx.active_dne_valid, 1, 0) AS active_dne_count,

    -- Rolling features
    r.pv_cap_7d,  r.pv_cap_30d,  r.pv_cap_60d,  r.pv_cap_90d,  r.pv_cap_all,
    r.pv_dne_7d,  r.pv_dne_30d,  r.pv_dne_60d,  r.pv_dne_90d,  r.pv_dne_all,

    r.walls_offer_cap_7d,  r.walls_offer_cap_30d,  r.walls_offer_cap_60d,  r.walls_offer_cap_90d,  r.walls_offer_cap_all,
    r.walls_offer_dne_7d,  r.walls_offer_dne_30d,  r.walls_offer_dne_60d,  r.walls_offer_dne_90d,  r.walls_offer_dne_all,

    r.nl_cap_7d,  r.nl_cap_30d,  r.nl_cap_60d,  r.nl_cap_90d,  r.nl_cap_all,
    r.nl_dne_7d,  r.nl_dne_30d,  r.nl_dne_60d,  r.nl_dne_90d,  r.nl_dne_all,

    r.subs_ended_cap_7d,  r.subs_ended_cap_30d,  r.subs_ended_cap_60d,  r.subs_ended_cap_90d,  r.subs_ended_cap_all,
    r.subs_ended_dne_7d,  r.subs_ended_dne_30d,  r.subs_ended_dne_60d,  r.subs_ended_dne_90d,  r.subs_ended_dne_all,
    r.subs_ended_other_7d,r.subs_ended_other_30d,r.subs_ended_other_60d,r.subs_ended_other_90d,r.subs_ended_other_all,

    -- Ratios (safe)
    IFNULL(r.pv_cap_7d  / NULLIF(r.pv_cap_30d, 0), 0)  AS pv_cap_ratio_7_30,
    IFNULL(r.pv_cap_30d / NULLIF(r.pv_cap_90d, 0), 0)  AS pv_cap_ratio_30_90,
    IFNULL(r.pv_dne_7d  / NULLIF(r.pv_dne_30d, 0), 0)  AS pv_dne_ratio_7_30,
    IFNULL(r.pv_dne_30d / NULLIF(r.pv_dne_90d, 0), 0)  AS pv_dne_ratio_30_90,

    IFNULL(r.walls_offer_cap_7d  / NULLIF(r.walls_offer_cap_30d, 0), 0) AS walls_cap_ratio_7_30,
    IFNULL(r.walls_offer_cap_30d / NULLIF(r.walls_offer_cap_90d, 0), 0) AS walls_cap_ratio_30_90,
    IFNULL(r.walls_offer_dne_7d  / NULLIF(r.walls_offer_dne_30d, 0), 0) AS walls_dne_ratio_7_30,
    IFNULL(r.walls_offer_dne_30d / NULLIF(r.walls_offer_dne_90d, 0), 0) AS walls_dne_ratio_30_90,

    IFNULL(r.nl_cap_7d  / NULLIF(r.nl_cap_30d, 0), 0) AS nl_cap_ratio_7_30,
    IFNULL(r.nl_cap_30d / NULLIF(r.nl_cap_90d, 0), 0) AS nl_cap_ratio_30_90,
    IFNULL(r.nl_dne_7d  / NULLIF(r.nl_dne_30d, 0), 0) AS nl_dne_ratio_7_30,
    IFNULL(r.nl_dne_30d / NULLIF(r.nl_dne_90d, 0), 0) AS nl_dne_ratio_30_90,

    IFNULL(r.subs_ended_cap_7d  / NULLIF(r.subs_ended_cap_30d, 0), 0) AS subs_cap_ratio_7_30,
    IFNULL(r.subs_ended_cap_30d / NULLIF(r.subs_ended_cap_90d, 0), 0) AS subs_cap_ratio_30_90,
    IFNULL(r.subs_ended_dne_7d  / NULLIF(r.subs_ended_dne_30d, 0), 0) AS subs_dne_ratio_7_30,
    IFNULL(r.subs_ended_dne_30d / NULLIF(r.subs_ended_dne_90d, 0), 0) AS subs_dne_ratio_30_90,

    -- RFV (by media)
    rp.recency_cap, rp.frequency_cap, rp.value_cap, rp.rfv_cap,
    rp.recency_dne, rp.frequency_dne, rp.value_dne, rp.rfv_dne,

    -- Money & failures (pre-joined)
    ocp.total_paid_so_far,
    ocp.total_unsuccessful_payments_so_far,

    -- Previous free sub flags (pre-joined)
    ff.had_prev_free_cap,
    ff.had_prev_free_dne,

    -- Total days previously subscribed (pre-joined)
    sdp.total_days_prev_cap,
    sdp.total_days_prev_dne,
    sdp.total_days_prev_other,

    -- Last payment outcome (pre-joined)
    lpp.last_payment_outcome,

    -- Keep active flags
    idx.active_cap_valid,
    idx.active_dne_valid

  FROM index_filtered idx
  LEFT JOIN rolling         r    ON r.lguserid = idx.lguserid AND r.date = idx.date
  LEFT JOIN rfv_pivot       rp   ON rp.lguserid = idx.lguserid AND rp.date = idx.date
  LEFT JOIN oc_pick         ocp  ON ocp.lguserid = idx.lguserid AND ocp.date = idx.date
  LEFT JOIN subs_days_pick  sdp  ON sdp.lguserid = idx.lguserid AND sdp.date = idx.date
  LEFT JOIN last_payment_pick lpp ON lpp.lguserid = idx.lguserid AND lpp.date = idx.date
  LEFT JOIN free_flags      ff   ON ff.lguserid = idx.lguserid AND ff.date = idx.date
)


-- 22) Join demographics (snapshot), leave imputation to Python
SELECT
  target_date AS scoring_date,
  f.* EXCEPT(candidate_cap, candidate_dne),
  d.education, d.workpos, d.sex,
  -- Missingness indicators for RFV (non-cumulative sparse vars) – handy in Python too
  IF(recency_cap IS NULL, 1, 0) AS miss_recency_cap,
  IF(recency_dne IS NULL, 1, 0) AS miss_recency_dne,
  IF(frequency_cap IS NULL, 1, 0) AS miss_frequency_cap,
  IF(frequency_dne IS NULL, 1, 0) AS miss_frequency_dne,
  IF(value_cap IS NULL, 1, 0) AS miss_value_cap,
  IF(value_dne IS NULL, 1, 0) AS miss_value_dne,
  IF(rfv_cap IS NULL, 1, 0) AS miss_rfv_cap,
  IF(rfv_dne IS NULL, 1, 0) AS miss_rfv_dne
FROM final f
LEFT JOIN `economedia-data-prod-laoy.public.lgusersdet` d
  ON d.lguserid = f.user_id
LEFT JOIN pv_users u
  ON u.lguserid = f.user_id
WHERE
  f.date = target_date
  AND NOT (f.candidate_cap AND (COALESCE(u.has_cap_pv, FALSE) = FALSE))
  AND NOT (f.candidate_dne AND (COALESCE(u.has_dne_pv, FALSE) = FALSE));
