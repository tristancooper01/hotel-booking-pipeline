-- Seasonal rate curve. Averaged across years so the shape is not a trend line.
SELECT arrival_month,
       CASE arrival_month
            WHEN 'January' THEN 1 WHEN 'February' THEN 2 WHEN 'March' THEN 3
            WHEN 'April' THEN 4 WHEN 'May' THEN 5 WHEN 'June' THEN 6
            WHEN 'July' THEN 7 WHEN 'August' THEN 8 WHEN 'September' THEN 9
            WHEN 'October' THEN 10 WHEN 'November' THEN 11 ELSE 12 END AS month_num,
       hotel_name,
       COUNT(*)                                  AS bookings,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate
FROM v_bookings
GROUP BY arrival_month, hotel_name
ORDER BY month_num, hotel_name;
