-- Do returning guests behave differently enough to justify a loyalty push?
SELECT CASE is_loyal_guest WHEN 1 THEN 'Repeat guest' ELSE 'First-time guest' END AS guest_type,
       COUNT(*)                                  AS bookings,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate,
       ROUND(AVG(total_nights), 2)               AS avg_nights,
       ROUND(AVG(lead_time), 1)                  AS avg_lead_time_days
FROM v_bookings
GROUP BY guest_type
ORDER BY bookings DESC;
