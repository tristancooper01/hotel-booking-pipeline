-- Demand, rate and revenue over the full calendar, one row per month.
SELECT arrival_year_month,
       COUNT(*)                                  AS bookings,
       SUM(is_canceled)                          AS cancellations,
       COUNT(*) - SUM(is_canceled)               AS stays,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate,
       ROUND(SUM(realized_revenue), 2)           AS realized_revenue,
       ROUND(SUM(lost_revenue), 2)               AS lost_revenue
FROM v_bookings
GROUP BY arrival_year_month
ORDER BY arrival_year_month;
