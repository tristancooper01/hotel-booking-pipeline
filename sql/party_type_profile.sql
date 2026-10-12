-- Who books, what they pay, and how long they stay.
SELECT party_type,
       COUNT(*)                                  AS bookings,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate,
       ROUND(AVG(total_nights), 2)               AS avg_nights,
       ROUND(AVG(booking_value), 2)              AS avg_booking_value,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct
FROM v_bookings
GROUP BY party_type
ORDER BY bookings DESC;
