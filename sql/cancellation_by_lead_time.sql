-- How far ahead a booking is made is the strongest single cancellation signal.
SELECT lead_time_bucket,
       COUNT(*)                                  AS bookings,
       SUM(is_canceled)                          AS cancellations,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate,
       ROUND(SUM(lost_revenue), 2)               AS lost_revenue
FROM v_bookings
GROUP BY lead_time_bucket
ORDER BY MIN(lead_time);
