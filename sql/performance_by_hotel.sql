-- The two properties side by side. City vs Resort is the dataset's main split.
SELECT hotel_name,
       COUNT(*)                                  AS bookings,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(lead_time), 1)                  AS avg_lead_time_days,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate,
       ROUND(AVG(total_nights), 2)               AS avg_nights,
       ROUND(SUM(realized_revenue), 2)           AS realized_revenue,
       ROUND(SUM(lost_revenue), 2)               AS lost_revenue,
       ROUND(AVG(is_loyal_guest) * 100, 2)       AS repeat_guest_pct
FROM v_bookings
GROUP BY hotel_name
ORDER BY bookings DESC;
