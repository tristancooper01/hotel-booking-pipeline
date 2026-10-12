-- Special requests are a proxy for guest intent: more requests, fewer walk-aways.
SELECT CASE WHEN total_of_special_requests >= 3 THEN '3+'
            ELSE CAST(total_of_special_requests AS TEXT) END AS special_requests,
       MIN(total_of_special_requests)            AS sort_key,
       COUNT(*)                                  AS bookings,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate
FROM v_bookings
GROUP BY special_requests
ORDER BY sort_key;
