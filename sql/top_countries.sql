-- Source markets, ranked by volume, with the risk each one carries.
SELECT country_code,
       COUNT(*)                                  AS bookings,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM v_bookings), 2) AS share_of_bookings_pct,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate,
       ROUND(AVG(total_nights), 2)               AS avg_nights,
       ROUND(SUM(realized_revenue), 2)           AS realized_revenue
FROM v_bookings
GROUP BY country_code
ORDER BY bookings DESC
LIMIT 15;
