-- Arrival-day pattern, used for the day-by-month demand heatmap.
SELECT arrival_day_name,
       arrival_month,
       COUNT(*)                                  AS bookings,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate
FROM v_bookings
GROUP BY arrival_day_name, arrival_month;
