-- Getting a different room than you reserved: how often, and does it track with
-- cancelling? (Rooms are assigned at check-in, so this is largely post-hoc.)
SELECT hotel_name,
       CASE room_changed WHEN 1 THEN 'Room changed' ELSE 'Room as reserved' END AS room_outcome,
       COUNT(*)                                  AS bookings,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate
FROM v_bookings
GROUP BY hotel_name, room_outcome
ORDER BY hotel_name, room_outcome;
