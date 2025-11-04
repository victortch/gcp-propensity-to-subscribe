# ------------------------
# Paste the BigQuery SQL here
# ------------------------
SQL_TEXT = """
-- TRAINING DATASET ASSEMBLY (Capital & Dnevnik) --------------------------------
-- Reads only once; returns user-date rows with labels + features

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
  WHERE DATE(a.createdate) >= @start_date
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
  WHERE DATE(created_ts) BETWEEN @start_date AND @freeze_date
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
  WHERE DATE(datetime) BETWEEN @start_date AND @freeze_date
    AND sid IN (1,2) AND COALESCE(lguserid,0) > 0
  GROUP BY 1,2,3
  UNION ALL
  SELECT lguserid, DATE(datetime) AS date, sid, COUNT(*) AS pv_count
  FROM `economedia-data-prod-laoy.public.paywall_log_archive_full`
  WHERE DATE(datetime) BETWEEN @start_date AND @freeze_date
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
  WHERE DATE(date) BETWEEN @start_date AND @freeze_date
    AND lguserid IS NOT NULL
),
walls_offer_daily AS (
  SELECT
    lguserid, date,
    SUM(CASE
          WHEN sid=1 AND (wall IN (2,3,5,6) OR (wall=1 AND date >= @cap_wall1_offer_start)) THEN 1 ELSE 0
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
  WHERE n.active = 1 AND n.sid IN (1,2) AND DATE(n.regdate) BETWEEN @start_date AND @freeze_date
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
  WHERE sid IN (1,2) AND DATE(date) BETWEEN @start_date AND @freeze_date
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
  WHERE date BETWEEN DATE_ADD(@start_date, INTERVAL 90 DAY) AND @freeze_date

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



-- 17) Apply inclusion logic & cleanups
index_filtered AS (
  SELECT
    ia.lguserid, ia.date,
    ia.active_cap_any, ia.active_cap_valid, ia.active_dne_any, ia.active_dne_valid,
    nf.next_cap_is_valid, nf.next_dne_is_valid
  FROM index_active ia
  JOIN next_flags nf USING (lguserid, date)
  WHERE
    -- At least one media is eligible on this date and NEXT for that media is not invalid

    (NOT ia.active_cap_valid AND NOT ia.active_cap_any AND (nf.next_cap_is_valid IS NULL OR nf.next_cap_is_valid = TRUE))
    OR
    (ia.date >= @dne_start AND NOT ia.active_dne_valid AND NOT ia.active_dne_any AND (nf.next_dne_is_valid IS NULL OR nf.next_dne_is_valid = TRUE))
    
),


-- 18) Labels via JOIN + aggregation (90 and 30 days only)
labels_counts AS (
  SELECT
    i.*,

    -- Capital: any valid future order within 90d?
    COUNTIF(s.media='capital' AND s.is_valid
            AND s.created_ts <= TIMESTAMP_ADD(TIMESTAMP(i.date), INTERVAL 90 DAY)) > 0 AS cap_in_90d,
    
    -- Capital: any valid future order within 30d?
    COUNTIF(s.media='capital' AND s.is_valid
            AND s.created_ts <= TIMESTAMP_ADD(TIMESTAMP(i.date), INTERVAL 30 DAY)) > 0 AS cap_in_30d,

    -- Dnevnik: any valid future order within 90d?
    COUNTIF(s.media='dnevnik' AND s.is_valid
            AND s.created_ts <= TIMESTAMP_ADD(TIMESTAMP(i.date), INTERVAL 90 DAY)) > 0 AS dne_in_90d,
    
    -- Dnevnik: any valid future order within 30d?
    COUNTIF(s.media='dnevnik' AND s.is_valid
            AND s.created_ts <= TIMESTAMP_ADD(TIMESTAMP(i.date), INTERVAL 30 DAY)) > 0 AS dne_in_30d

  FROM index_filtered i
  LEFT JOIN subs_orders_sorted s
    ON s.lguserid = i.lguserid
   AND s.is_valid = TRUE
   AND s.created_ts > TIMESTAMP(i.date)
   AND s.media IN ('capital','dnevnik')
  GROUP BY
    i.lguserid, i.date,
    i.active_cap_any, i.active_cap_valid, i.active_dne_any, i.active_dne_valid,
    i.next_cap_is_valid, i.next_dne_is_valid
),


labels AS (
  SELECT
    lc.* EXCEPT(cap_in_90d, dne_in_90d, cap_in_30d, dne_in_30d),

    -- Capital 90d label with NA masking for active/censored
    CASE
      WHEN lc.active_cap_any OR lc.date > DATE_SUB(@freeze_date, INTERVAL 90 DAY)
      THEN NULL ELSE lc.cap_in_90d
    END AS y_cap_90d,
    
    -- Capital 30d label with NA masking for active/censored
    CASE
      WHEN lc.active_cap_any OR lc.date > DATE_SUB(@freeze_date, INTERVAL 30 DAY)
      THEN NULL ELSE lc.cap_in_30d
    END AS y_cap_30d,

    -- Dnevnik 90d label with NA masking for pre-start/active/censored
    CASE
      WHEN lc.date < @dne_start OR lc.active_dne_any OR lc.date > DATE_SUB(@freeze_date, INTERVAL 90 DAY)
      THEN NULL ELSE lc.dne_in_90d
    END AS y_dne_90d,
    
    -- Dnevnik 30d label with NA masking for pre-start/active/censored
    CASE
      WHEN lc.date < @dne_start OR lc.active_dne_any OR lc.date > DATE_SUB(@freeze_date, INTERVAL 30 DAY)
      THEN NULL ELSE lc.dne_in_30d
    END AS y_dne_30d
    
  FROM labels_counts lc
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
    l.lguserid, l.date,
    oc.cum_paid_amount, oc.cum_fail_cnt,
    ROW_NUMBER() OVER (
      PARTITION BY l.lguserid, l.date
      ORDER BY oc.date DESC
    ) AS rn
  FROM labels l
  LEFT JOIN orders_cum oc
    ON oc.lguserid = l.lguserid
   AND oc.date     <= l.date
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
    l.lguserid, l.date, s.media, s.cum_days_covered_upto_date,
    ROW_NUMBER() OVER (
      PARTITION BY l.lguserid, l.date, s.media
      ORDER BY s.date DESC
    ) AS rn
  FROM labels l
  LEFT JOIN subs_days_cum s
    ON s.lguserid = l.lguserid
   AND s.date     <  l.date
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
    l.lguserid, l.date,
    CASE WHEN o.state = 1 THEN 2 ELSE 1 END AS last_payment_outcome,
    ROW_NUMBER() OVER (
      PARTITION BY l.lguserid, l.date
      ORDER BY o.created_ts DESC
    ) AS rn
  FROM labels l
  LEFT JOIN orders_raw o
    ON o.lguserid = l.lguserid
   AND DATE(o.created_ts) <= l.date
),
last_payment_pick AS (
  SELECT lguserid, date, IFNULL(last_payment_outcome, 0) AS last_payment_outcome
  FROM last_payment_latest
  WHERE rn = 1
),

-- Ever free subscription before index date (per media)
free_flags AS (
  SELECT
    l.lguserid, l.date,
    -- Exists any free sub earlier?
    IF(COUNTIF(f.media='capital')>0, 1, 0) AS had_prev_free_cap,
    IF(COUNTIF(f.media='dnevnik')>0, 1, 0) AS had_prev_free_dne
  FROM labels l
  LEFT JOIN free_sub_ever f
    ON f.lguserid = l.lguserid
   AND f.first_free_date < l.date
  GROUP BY l.lguserid, l.date
),


-- 21) Final assembly: labels + features (join rolling + rfv + orders cum + active flags + totals)
final AS (
  SELECT
    l.lguserid AS user_id,
    l.date,

    -- Labels (nullable INT64)

    CAST(l.y_cap_90d AS INT64) AS y_cap_90d,
    CAST(l.y_cap_30d AS INT64) AS y_cap_30d,  
    CAST(l.y_dne_90d AS INT64) AS y_dne_90d,
    CAST(l.y_dne_30d AS INT64) AS y_dne_30d, 

    -- Current active counts per media (0/1)
    IF(l.active_cap_valid, 1, 0) AS active_cap_count,
    IF(l.active_dne_valid, 1, 0) AS active_dne_count,

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
    l.active_cap_valid, l.active_dne_valid

  FROM labels l
  LEFT JOIN rolling         r    ON r.lguserid = l.lguserid AND r.date = l.date
  LEFT JOIN rfv_pivot       rp   ON rp.lguserid = l.lguserid AND rp.date = l.date
  LEFT JOIN oc_pick         ocp  ON ocp.lguserid = l.lguserid AND ocp.date = l.date
  LEFT JOIN subs_days_pick  sdp  ON sdp.lguserid = l.lguserid AND sdp.date = l.date
  LEFT JOIN last_payment_pick lpp ON lpp.lguserid = l.lguserid AND lpp.date = l.date
  LEFT JOIN free_flags      ff   ON ff.lguserid = l.lguserid AND ff.date = l.date
)


-- 22) Join demographics (snapshot), leave imputation to Python
SELECT
  f.*,
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
  -- keep the universal tail de-censoring
  f.date <= DATE_SUB(@freeze_date, INTERVAL 90 DAY)
  -- DROP rows that would have been candidates for Cap label, if user never had any Capital PV
  AND NOT ( (f.y_cap_90d IS NOT NULL OR f.y_cap_30d IS NOT NULL) AND (COALESCE(u.has_cap_pv, FALSE) = FALSE) )
  -- DROP rows that would have been candidates for Dne label, if user never had any Dnevnik PV
  AND NOT ( (f.y_dne_90d IS NOT NULL OR f.y_dne_30d IS NOT NULL) AND (COALESCE(u.has_dne_pv, FALSE) = FALSE) )
ORDER BY user_id, date;
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Economedia - Recommendation Engine
# CV build script with eval balancing + per-run tagging + metadata capture.
#
# - Training remains identical (composition, balancing).
# - Validation/Test: keep all positives, downsample negatives to match #positives.
# - Capture and upload the same metadata you print today (drop summaries + natural split counts).
# - Append-only writes with RUN_ID tagging; delete rows for the same RUN_ID first for idempotency.
# - Train table: stores ONLY run_id (no other per-run fields). All other per-run data in metadata.
# - SQL_TEXT is imported (not rewritten) from utils.bqutils.

#from utils.bqutils import SQL_TEXT

import os
import sys
import gc
import tempfile
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple

import uuid
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, LoadJobConfig, SchemaUpdateOption
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound
import re


# Optional: BigQuery Storage for faster downloads (falls back if missing)
try:
    from google.cloud import bigquery_storage_v1
    _HAS_BQSTORAGE = True
except Exception:
    bigquery_storage_v1 = None
    _HAS_BQSTORAGE = False


# =========================
# Configuration
# =========================
SERVICE_ACCOUNT_KEY = os.getenv(
    "SERVICE_ACCOUNT_KEY",
    "/Users/victortchervenobrejki/Documents/Economedia/economedia-data-prod-laoy-2cd58f972a4e.json",
)
PROJECT_ID   = os.getenv("PROJECT_ID", "economedia-data-prod-laoy")
BQ_LOCATION  = os.getenv("BQ_LOCATION", "europe-west3")

# BigQuery targets
BQ_DATASET    = os.getenv("BQ_DATASET", "propensity_to_subscribe")
BQ_TABLE      = f"{PROJECT_ID}.{BQ_DATASET}.train_data"            # final consolidated rows
BQ_META_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.cv_build_metadata"     # metadata (counts, drops, run info)

# Preview / upload toggle
DRY_RUN = False   # True = preview sizes & skip upload; False = upload to BigQuery
_env = os.getenv("DRY_RUN")
if _env is not None:
    DRY_RUN = _env not in ("0","false","False","no","NO")

# Training window / business dates
START_DATE  = date(2024, 10, 1)
FREEZE_DATE = date.today() - timedelta(days=7)  # use your current global freeze
DNE_START   = date(2024, 10, 1)
CAP_WALL1_OFFER_START = date(2024, 10, 1)

# CV & sampling knobs
TEST_USER_FRAC   = 0.20         # share of users held out for final test
N_FOLDS          = 5            # number of user-grouped folds on Dev users
EMBARGO_DAYS     = 0            # 0 is fine for user-disjoint, per-user features
BALANCE_TRAINING = True         # 50/50 positive/negative via downsampling after one-per-streak
RANDOM_STATE     = 42
SPLIT_DATE       = pd.Timestamp(FREEZE_DATE) - pd.Timedelta(days=120)  # global time cutoff for CV folds

# Label configs: (y_column_name, short_tag)
LABELS: List[Tuple[str, str]] = [
    #("y_cap_90d", "cap_90d"),
    #("y_dne_90d", "dne_90d"),
    ("y_cap_30d", "cap_30d"),
    ("y_dne_30d", "dne_30d"),
]

# =========================
# Run fingerprint (for traceability across repeated uploads)
# =========================
RUN_ID = os.getenv("RUN_ID") or (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
)
RUN_STARTED_AT = datetime.now(timezone.utc)  # timezone-aware UTC

# Optional: attach code/SQL provenance (handy in joins and audits)
GIT_SHA = os.getenv("GIT_SHA", "")  # set this in CI or locally if you want
try:
    SQL_SHA = hashlib.sha1(SQL_TEXT.encode("utf-8")).hexdigest()[:12]
except Exception:
    SQL_SHA = ""



def label_safe(value: str) -> str:
    """Make a string compliant with GCP label value rules."""
    v = value.lower()
    v = re.sub(r'[^a-z0-9_-]+', '-', v)  # keep only a-z0-9_-
    v = v.strip('-_')                     # trim ends
    if not v or not v[0].isalpha():       # must start with letter
        v = 'r' + v
    return v[:63]                         # max 63 chars

# =========================
# BigQuery I/O
# =========================
def make_bq_clients():
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_KEY, scopes=scopes
    )
    bq = bigquery.Client(project=PROJECT_ID, credentials=creds, location=BQ_LOCATION)
    bqs = None
    if _HAS_BQSTORAGE:
        try:
            bqs = bigquery_storage_v1.BigQueryReadClient(credentials=creds)
        except Exception:
            bqs = None
    return bq, bqs


def ensure_dataset():
    bq, _ = make_bq_clients()
    bq.create_dataset(BQ_DATASET, exists_ok=True)


def table_exists(table_fqn: str) -> bool:
    bq, _ = make_bq_clients()
    try:
        bq.get_table(table_fqn)
        return True
    except NotFound:
        return False


def delete_existing_run_rows(table_fqn: str, run_id: str):
    """Idempotency: remove any previous rows for this RUN_ID before appending."""
    if not table_exists(table_fqn):
        return
    bq, _ = make_bq_clients()
    q = f"DELETE FROM `{table_fqn}` WHERE run_id = @run_id"
    job = bq.query(
        q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
        ),
    )
    job.result()
    print(f"[info] Deleted existing rows for run_id={run_id} in {table_fqn}")


def load_base_dataframe() -> pd.DataFrame:
    """Execute SQL_TEXT and return a Pandas DataFrame with light post-processing."""
    bq, bqs = make_bq_clients()
    job_config = QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("start_date", "DATE", START_DATE),
        bigquery.ScalarQueryParameter("freeze_date", "DATE", FREEZE_DATE),
        bigquery.ScalarQueryParameter("dne_start", "DATE", DNE_START),
        bigquery.ScalarQueryParameter("cap_wall1_offer_start", "DATE", CAP_WALL1_OFFER_START),
    ])
    job = bq.query(SQL_TEXT, job_config=job_config)
    res = job.result()
    df = res.to_dataframe(bqstorage_client=bqs) if bqs is not None else res.to_dataframe()

    # Basic post-processing
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Ensure label dtypes are nullable ints
    for c in [c for c in df.columns if c.startswith("y_")]:
        df[c] = df[c].astype("Int32")  # pandas nullable int

    # RFV: fill NAs with 0 (keep missing flags from SQL if present)
    for c in ["recency_cap","frequency_cap","value_cap","rfv_cap",
              "recency_dne","frequency_dne","value_dne","rfv_dne"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Demographics: keep numeric categories; add simple missing flags
    for c in ["education", "workpos", "sex"]:
        if c in df.columns:
            s = df[c].astype("string")
            miss = s.isna() | s.str.strip().eq("")
            token = s.str.extract(r"(-?\d+)")[0]
            num = pd.to_numeric(token, errors="coerce")
            df[c] = num.fillna(0).astype("Int32")
            df[f"{c}_missing"] = miss.astype("Int8")
            df[f"{c}_invalid"] = ((~miss) & num.isna()).astype("Int8")

    # Last payment outcome as nullable int
    if "last_payment_outcome" in df.columns:
        df["last_payment_outcome"] = df["last_payment_outcome"].astype("Int32")

    # Basic sanity
    required = {"user_id", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Base DF missing required columns: {missing}")

    return df


# =========================
# CV & Sampling Utilities
# =========================
def add_streak_ids(df: pd.DataFrame, y_col: str, user_col: str = "user_id") -> pd.Series:
    """
    Robust run-length-encoding of streaks per user for the given y column.
    Returns a Series aligned to df's index.
    """
    z = df.sort_values([user_col, "date"]).copy()
    y = z[y_col]
    if isinstance(y, pd.DataFrame):  # defensive
        y = y.iloc[:, 0]
    streak = (
        y.groupby(z[user_col])
         .apply(lambda x: (x != x.shift()).cumsum())
         .reset_index(level=0, drop=True)
    )
    streak.index = z.index
    return streak.reindex(df.index)


def one_per_streak(df: pd.DataFrame, user_col: str = "user_id") -> pd.DataFrame:
    """Pick exactly one row per (user, streak_id) deterministically (first by date)."""
    key = [user_col, "streak_id", "date"]
    return (df.sort_values(key)
              .groupby([user_col, "streak_id"], as_index=False)
              .first())


def summarize_split(df: pd.DataFrame, title: str) -> Dict[str, int]:
    """
    Print a human-friendly summary (unchanged) and return a dict with counts
    that can be captured in metadata (without pos_rate).
    """
    if df.empty:
        print(f"{title:30s} -> EMPTY")
        return {"n": 0, "pos": 0, "neg": 0, "users": 0}

    counts = df["y"].value_counts()
    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))
    total = pos + neg
    users = df["user_id"].nunique()
    pos_rate = (pos / total) if total > 0 else 0.0  # only for printing

    print(f"{title:30s} -> n={total:7d} | pos={pos:6d} neg={neg:6d} | pos_rate={pos_rate:6.2%} | users={users}")
    return {"n": total, "pos": pos, "neg": neg, "users": users}


def drop_consecutive_duplicates(df: pd.DataFrame,
                                group_col: str = "user_id",
                                order_col: str = "date",
                                exclude_cols: set | None = None) -> pd.DataFrame:
    """
    Collapse consecutive rows per user where *all other columns* (excluding `exclude_cols`)
    are identical. Keeps the first row of each identical run.

    - `exclude_cols` should at least include the ordering column (e.g. {'date'}).
    - The dependent variable (e.g. 'y') is kept in the signature by default so
      equality requires same label too.
    """
    if exclude_cols is None:
        exclude_cols = {order_col}
    else:
        exclude_cols = set(exclude_cols) | {order_col}

    # Ensure sorted by group + time
    df = df.sort_values([group_col, order_col]).copy()

    # Columns that define the "state" of a row (everything except group/id and excluded cols)
    sig_cols = [c for c in df.columns if c not in exclude_cols and c != group_col]

    if not sig_cols:
        return df  # nothing to compare

    # Build a stable per-row hash signature across sig_cols
    sig = pd.util.hash_pandas_object(df[sig_cols], index=False)

    # Compare with previous row's signature within each user
    prev_sig = sig.groupby(df[group_col]).shift()

    keep_mask = (prev_sig.isna()) | (sig.ne(prev_sig))
    # keep first row in each group/run, drop only exact consecutive duplicates
    return df.loc[keep_mask].copy()


def balance_eval_set(df: pd.DataFrame, *, random_state: int) -> pd.DataFrame:
    """
    Validation/Test balancing: keep all positives; sample negatives to match #positives.
    Sampling is without replacement when possible, otherwise with replacement.
    """
    if df.empty:
        return df.copy()

    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]

    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0:
        # Strict: if 0 positives, pick 0 negatives -> empty set
        return df.iloc[0:0].copy()

    replace = n_pos > n_neg
    neg_sampled = neg.sample(n=n_pos, replace=replace, random_state=random_state) if n_neg > 0 else neg.iloc[0:0]

    out = pd.concat([pos, neg_sampled], axis=0).sort_values(["user_id", "date"])
    return out


# ----- Metadata helpers

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def meta_drop_record(y_col: str, tag: str, before_len: int, after_len: int) -> Dict:
    """Metadata row for the duplicate-drop line."""
    dropped = before_len - after_len
    return {
        "run_id": RUN_ID,
        "run_started_at": RUN_STARTED_AT.isoformat(),
        "git_sha": GIT_SHA,
        "sql_sha": SQL_SHA,
        "y_col": y_col,
        "label_tag": tag,
        "split": "prep",
        "fold": pd.NA,
        "n": pd.NA,
        "pos": pd.NA,
        "neg": pd.NA,
        "users": pd.NA,
        "duplicates_dropped": int(dropped),
        "kept_after_drop": int(after_len),
        "created_at": _now_utc_iso(),
        "start_date": str(START_DATE),
        "freeze_date": str(FREEZE_DATE),
        "dne_start": str(DNE_START),
        "cap_wall1_offer_start": str(CAP_WALL1_OFFER_START),
        "random_state": int(RANDOM_STATE),
    }


def meta_split_record(y_col: str, tag: str, split: str, fold: int, counts: Dict[str, int]) -> Dict:
    """Metadata row for the TRAIN/VALID/TEST summary line (without pos_rate)."""
    return {
        "run_id": RUN_ID,
        "run_started_at": RUN_STARTED_AT.isoformat(),
        "git_sha": GIT_SHA,
        "sql_sha": SQL_SHA,
        "y_col": y_col,
        "label_tag": tag,
        "split": split,               # "train" | "val" | "test"
        "fold": int(fold),            # 0 for test
        "n": int(counts.get("n", 0)),
        "pos": int(counts.get("pos", 0)),
        "neg": int(counts.get("neg", 0)),
        "users": int(counts.get("users", 0)),
        "duplicates_dropped": pd.NA,
        "kept_after_drop": pd.NA,
        "created_at": _now_utc_iso(),
        "start_date": str(START_DATE),
        "freeze_date": str(FREEZE_DATE),
        "dne_start": str(DNE_START),
        "cap_wall1_offer_start": str(CAP_WALL1_OFFER_START),
        "random_state": int(RANDOM_STATE),
    }


def upload_meta_to_bq(records: List[Dict]):
    """
    Upload the accumulated metadata records to BigQuery (append-only).
    Caller is responsible for delete_existing_run_rows() idempotency per RUN_ID.
    """
    if not records:
        print("[info] No metadata records to upload.")
        return

    bq, _ = make_bq_clients()
    ensure_dataset()

    df = pd.DataFrame.from_records(records)

    # Stable schema dtypes
    if "fold" in df.columns:
        df["fold"] = df["fold"].astype("Int64")
    for col in ["n","pos","neg","users","duplicates_dropped","kept_after_drop","random_state"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    for col in ["run_id","git_sha","sql_sha","y_col","label_tag","split","created_at",
                "start_date","freeze_date","dne_start","cap_wall1_offer_start"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if "run_started_at" in df.columns:
        df["run_started_at"] = pd.to_datetime(df["run_started_at"])
    
    safe = label_safe(RUN_ID)
    job_config = LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema_update_options=[SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    job_config.labels = {"run_id": safe, "dest": "cv_build_metadata"}


    job = bq.load_table_from_dataframe(df, BQ_META_TABLE, job_config=job_config)
    job.result()
    print(f"[info] Uploaded {len(df):,} metadata rows (append) to {BQ_META_TABLE}")


# =========================
# CV Preparation (returns natural TEST + per-fold TRAIN/VAL) + metadata capture
# =========================
def prepare_cv_global(df_all: pd.DataFrame, y_col: str, *, label_tag: str) -> Tuple[pd.DataFrame, List[Tuple[int, pd.DataFrame, pd.DataFrame]], List[Dict]]:
    """
    Build user-disjoint dev/test splits, then GroupKFold over dev users.
    For each fold:
      - Train: dev users not in fold, dates <= SPLIT_DATE - EMBARGO, one-per-streak, optional 50/50 balance.
      - Val:   fold users, dates >  SPLIT_DATE + EMBARGO, natural (unbalanced).
    Returns: (test_df_natural, folds_list, metadata_records)
      - metadata_records captures the *printed* info (drop summary, per-split counts) WITHOUT pos_rate.
    """
    meta: List[Dict] = []

    df = df_all.copy()
    if "y" in df.columns:
        df = df.drop(columns=["y"])
    df = df[df[y_col].notna()].copy()
    df["y"] = df[y_col].astype(int)

    # Drop ALL original y_* columns now to avoid any duplicate 'y' later
    df = df.drop(columns=[c for c in df.columns if c.startswith("y_")], errors="ignore")

    # drop consecutive duplicates per user where every non-date column (incl. 'y') is identical
    before = len(df)
    df = drop_consecutive_duplicates(df, group_col="user_id", order_col="date", exclude_cols={"date"})
    after = len(df)
    if after < before:
        print(f"[info] {y_col}: dropped {before - after:,} consecutive duplicate rows (kept {after:,})")
    else:
        print(f"[info] {y_col}: dropped 0 consecutive duplicate rows (kept {after:,})")
    # capture metadata for the drop summary
    meta.append(meta_drop_record(y_col, label_tag, before_len=before, after_len=after))

    # Guard against duplicate column names anywhere
    assert df.columns.to_series().groupby(df.columns).size().max() == 1, \
        f"Duplicate column names detected: {df.columns[df.columns.duplicated()].tolist()}"

    # ---- User-level split (convert to ndarray to avoid IntegerArray shuffle warning)
    users = df["user_id"].dropna().to_numpy(dtype=np.int64, copy=True)
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(users)
    n_test = int(len(users) * TEST_USER_FRAC)
    test_users = set(users[:n_test])
    dev_users  = set(users[n_test:])

    test_df = df[df["user_id"].isin(test_users)].copy()  # natural distribution
    dev_df  = df[df["user_id"].isin(dev_users)].copy()

    # Summarize & capture natural TEST counts (printed like today)
    test_counts = summarize_split(test_df, f"{y_col} TEST")
    meta.append(meta_split_record(y_col, label_tag, "test", 0, test_counts))

    # ---- GroupKFold on Dev users
    n_groups = max(2, len(dev_df["user_id"].unique()))
    gkf = GroupKFold(n_splits=min(N_FOLDS, n_groups))

    folds: List[Tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for k, (_, val_idx) in enumerate(gkf.split(dev_df, groups=dev_df["user_id"])):
        val_users  = set(dev_df.iloc[val_idx]["user_id"].unique())
        train_users = dev_users - val_users

        train_cut = SPLIT_DATE - pd.Timedelta(days=EMBARGO_DAYS)
        val_cut   = SPLIT_DATE + pd.Timedelta(days=EMBARGO_DAYS)

        tr = dev_df[(dev_df["user_id"].isin(train_users)) & (dev_df["date"] <= train_cut)].copy()
        va = dev_df[(dev_df["user_id"].isin(val_users))   & (dev_df["date"] >  val_cut)].copy()

        if tr.empty or va.empty:
            print(f"[warn] {y_col} fold {k+1}: empty train/val after filters; skipping.")
            continue

        # One-per-streak for both classes (deterministic first-by-date)
        tr["streak_id"] = add_streak_ids(tr, "y", user_col="user_id")
        pos = one_per_streak(tr[tr["y"] == 1])
        neg = one_per_streak(tr[tr["y"] == 0])

        if BALANCE_TRAINING:
            n = min(len(pos), len(neg))
            if n == 0:
                print(f"[warn] {y_col} fold {k+1}: no pos/neg after filtering; skipping.")
                continue
            if len(pos) > n: pos = pos.sample(n, random_state=RANDOM_STATE)
            if len(neg) > n: neg = neg.sample(n, random_state=RANDOM_STATE)

        tr_final = (pd.concat([pos, neg], axis=0)
                      .sort_values(["user_id","date"])
                      .drop(columns=["streak_id"], errors="ignore"))

        # VAL: keep natural distribution; no streak collapsing, no balancing
        va_final = va.copy()

        # Safety: ensure no user leakage
        assert set(tr_final["user_id"]).isdisjoint(set(va_final["user_id"])), \
            f"User leakage in fold {k+1} for {y_col}"

        # Print & capture TRAIN/VAL counts (as you do today)
        tr_counts = summarize_split(tr_final, f"{y_col} TRAIN f{k+1}")
        va_counts = summarize_split(va_final, f"{y_col} VALID f{k+1}")
        meta.append(meta_split_record(y_col, label_tag, "train", k+1, tr_counts))
        meta.append(meta_split_record(y_col, label_tag, "val",   k+1, va_counts))

        folds.append((k+1, tr_final, va_final))

    return test_df, folds, meta


# =========================
# Upload & Preview helpers (streaming)
# =========================
def _bytes_human(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def preview_dataframe_size(df: pd.DataFrame, label: str = "consolidated") -> Dict[str, int]:
    """
    Returns a dict with row count, column count, RAM bytes, and an approximate
    on-disk size by writing a temporary Parquet file (snappy).
    """
    info = {}
    info["rows"] = len(df)
    info["cols"] = df.shape[1]
    info["ram_bytes"] = int(df.memory_usage(deep=True).sum())

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"{label}.parquet"
        df.to_parquet(p, index=False)  # requires pyarrow or fastparquet
        info["parquet_bytes"] = p.stat().st_size

    print(f"[preview] {label}: rows={info['rows']:,}, cols={info['cols']:,}")
    print(f"[preview] RAM footprint (pandas): {_bytes_human(info['ram_bytes'])}")
    print(f"[preview] Parquet (snappy) size:   {_bytes_human(info['parquet_bytes'])}")
    print("          (Parquet size is a rough proxy for BigQuery stored size.)")
    return info


def preview_cv_mix(df: pd.DataFrame):
    """Small sanity table of label/split/fold counts for the given slice."""
    mix = (df.groupby(["label","split","fold"])
             .agg(rows=("user_id","size"), users=("user_id","nunique"))
             .reset_index()
             .sort_values(["label","split","fold"]))
    print("\n[preview] label/split/fold breakdown:")
    print(mix.to_string(index=False))
    return mix


def tag_with_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach only the per-run identifier to the train data rows.
    All other per-run metadata lives in cv_build_metadata, joined by run_id.
    """
    df = df.copy()
    df["run_id"] = RUN_ID
    return df


def upload_df_to_bq(df: pd.DataFrame):
    """
    Upload one slice to BigQuery (append-only).
    Caller is responsible for delete_existing_run_rows() idempotency per RUN_ID.
    """
    bq, _ = make_bq_clients()
    ensure_dataset()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Casts for stable schema
    if "y" in df.columns:
        df["y"] = df["y"].astype("Int64")
    if "fold" in df.columns:
        df["fold"] = df["fold"].astype("Int64")
    for c in ("label","split","run_id"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    safe = label_safe(RUN_ID)
    job_config = LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema_update_options=[SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    job_config.labels = {"run_id": safe, "dest": "train_data"}


    job = bq.load_table_from_dataframe(df, BQ_TABLE, job_config=job_config)
    job.result()
    print(f"[info] Uploaded chunk: {len(df):,} rows (append) to {BQ_TABLE}")


# =========================
# Main (streaming to avoid OOM)
# =========================
def main():
    print(f"[info] RUN_ID={RUN_ID} | started_at={RUN_STARTED_AT.isoformat()} | git_sha={GIT_SHA} | sql_sha={SQL_SHA}")

    # 0) Idempotency: ensure dataset and delete any previous rows for this RUN_ID
    if not DRY_RUN:
        ensure_dataset()
        delete_existing_run_rows(BQ_TABLE, RUN_ID)
        delete_existing_run_rows(BQ_META_TABLE, RUN_ID)

    # 1) Load base dataset from BigQuery
    df_train = load_base_dataframe()
    print(f"[info] Base DataFrame: {len(df_train):,} rows, {df_train['user_id'].nunique():,} users, dates {df_train['date'].min().date()}..{df_train['date'].max().date()}")

    uploaded_rows_total = 0
    meta_records_all: List[Dict] = []

    # 2) Build and handle CV datasets per label in a streaming manner
    for y_col, tag in LABELS:
        print(f"\n=== Preparing label: {y_col} ({tag}) ===")

        # Prepare natural TEST and per-fold TRAIN/VAL (TRAIN balanced; VAL natural)
        test_df_nat, folds, meta_records = prepare_cv_global(df_train, y_col, label_tag=tag)
        meta_records_all.extend(meta_records)

        # ---- TEST slice: balance negatives to match positives (keep all positives)
        te_nat = test_df_nat.copy()  # natural distribution (metadata already captured)
        te_bal = balance_eval_set(te_nat, random_state=RANDOM_STATE)
        te = te_bal.copy()
        te["label"], te["split"], te["fold"] = tag, "test", 0
        te = tag_with_run(te)

        preview_cv_mix(te)
        preview_dataframe_size(te, label=f"{tag}_test_balanced")

        if not DRY_RUN:
            upload_df_to_bq(te)
            uploaded_rows_total += len(te)

        del te, te_bal, te_nat
        gc.collect()

        # ---- TRAIN/VAL folds
        for fold_id, tr_df, va_df in folds:
            # ---------- TRAIN (unchanged: identical to current production)
            tr = tr_df.copy()
            tr["label"], tr["split"], tr["fold"] = tag, "train", fold_id
            tr = tag_with_run(tr)

            preview_cv_mix(tr)
            preview_dataframe_size(tr, label=f"{tag}_train_f{fold_id}")

            if not DRY_RUN:
                upload_df_to_bq(tr)
                uploaded_rows_total += len(tr)

            del tr
            gc.collect()

            # ---------- VALIDATION: balance negatives to match positives (keep all positives)
            va_nat = va_df.copy()  # natural distribution (metadata already captured)
            va_bal = balance_eval_set(va_nat, random_state=RANDOM_STATE)
            va = va_bal.copy()
            va["label"], va["split"], va["fold"] = tag, "val", fold_id
            va = tag_with_run(va)

            preview_cv_mix(va)
            preview_dataframe_size(va, label=f"{tag}_val_f{fold_id}_balanced")

            if not DRY_RUN:
                upload_df_to_bq(va)
                uploaded_rows_total += len(va)

            del va, va_bal, va_nat
            gc.collect()

        # Upload metadata for this label (append)
        if DRY_RUN:
            if meta_records:
                print("\n[preview] Metadata rows (natural counts + drop summaries):")
                mdf = pd.DataFrame(meta_records).sort_values(["y_col","split","fold"])
                print(mdf.to_string(index=False))
        else:
            upload_meta_to_bq(meta_records)

        del folds, test_df_nat, meta_records
        gc.collect()

    if DRY_RUN:
        print("\n[info] DRY_RUN is True -> no data uploaded.")
        if meta_records_all:
            mdf_all = pd.DataFrame(meta_records_all).sort_values(["y_col","split","fold"])
            print("\n[preview] ALL metadata rows (across labels):")
            print(mdf_all.to_string(index=False))
    else:
        print(f"\n[info] Total uploaded rows: {uploaded_rows_total:,} to {BQ_TABLE}")
        print(f"[info] Metadata written to {BQ_META_TABLE}")
        print(f"[info] Completed RUN_ID={RUN_ID}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[error]", e)
        sys.exit(1)
