-- Headline numbers for the whole reporting period.
SELECT COUNT(*)                                  AS total_bookings,
       SUM(is_canceled)                          AS total_cancellations,
       ROUND(AVG(is_canceled) * 100, 2)          AS cancellation_rate_pct,
       ROUND(AVG(lead_time), 1)                  AS avg_lead_time_days,
       ROUND(AVG(adr), 2)                        AS avg_daily_rate,
       ROUND(AVG(total_nights), 2)               AS avg_nights,
       ROUND(AVG(total_guests), 2)               AS avg_guests,
       ROUND(SUM(realized_revenue), 2)           AS realized_revenue,
       ROUND(SUM(lost_revenue), 2)               AS lost_revenue,
       -- 'Unknown' is the label for a missing country, not a source market.
       COUNT(DISTINCT CASE WHEN country_code <> 'Unknown'
                           THEN country_code END) AS countries,
       MIN(arrival_date)                         AS period_start,
       MAX(arrival_date)                         AS period_end
FROM v_bookings;
