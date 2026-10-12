-- The counter-intuitive one: non-refundable deposits cancel far MORE, not less.
SELECT deposit_type,
       COUNT(*)                                  AS bookings,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(lead_time), 1)                  AS avg_lead_time_days,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate
FROM v_bookings
GROUP BY deposit_type
ORDER BY bookings DESC;
