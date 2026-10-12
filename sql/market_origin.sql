-- Domestic versus international demand. The headline cancellation rate is two
-- very different populations averaged together, and this is the split that
-- explains most of the other patterns in the data.
SELECT CASE WHEN country_code = 'PRT' THEN 'Portugal (domestic)'
            WHEN country_code = 'Unknown' THEN 'Unknown origin'
            ELSE 'International' END         AS origin,
       COUNT(*)                              AS bookings,
       ROUND(AVG(is_canceled) * 100, 2)      AS cancellation_rate_pct,
       ROUND(AVG(adr), 2)                    AS avg_daily_rate,
       ROUND(AVG(lead_time), 1)              AS avg_lead_time_days,
       ROUND(SUM(realized_revenue), 2)       AS realized_revenue,
       ROUND(SUM(lost_revenue), 2)           AS lost_revenue,
       -- Euros of booked value lost for every euro actually earned.
       ROUND(SUM(lost_revenue) / SUM(realized_revenue), 2) AS lost_per_eur_earned
FROM v_bookings
GROUP BY origin
ORDER BY bookings DESC;
