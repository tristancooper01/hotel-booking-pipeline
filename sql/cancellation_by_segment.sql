-- Which acquisition channels bring demand that actually shows up.
SELECT market_segment,
       COUNT(*)                                  AS bookings,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM v_bookings), 2) AS share_of_bookings_pct,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate,
       ROUND(AVG(lead_time), 1)                  AS avg_lead_time_days,
       ROUND(SUM(realized_revenue), 2)           AS realized_revenue,
       ROUND(SUM(lost_revenue), 2)               AS lost_revenue
FROM v_bookings
GROUP BY market_segment
ORDER BY bookings DESC;
