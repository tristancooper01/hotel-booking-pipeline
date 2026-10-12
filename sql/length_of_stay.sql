-- Stay-length distribution, capped into a tail bucket so one 60-night outlier
-- does not stretch the axis.
SELECT CASE WHEN total_nights = 0 THEN '0 nights'
            WHEN total_nights >= 15 THEN '15+ nights'
            ELSE CAST(total_nights AS TEXT) || ' nights' END AS stay_length,
       MIN(total_nights)                         AS sort_key,
       COUNT(*)                                  AS bookings,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate
FROM v_bookings
GROUP BY stay_length
ORDER BY sort_key;
