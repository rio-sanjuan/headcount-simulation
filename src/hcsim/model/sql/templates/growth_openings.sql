-- Growth openings: growth requisitions of the population opened per cost center and month,
-- keyed by month-end in ($start, $end], for every cost center with someone of the population
-- in its stock at that month-end. A cost center that opened none counts zero, and a month it
-- staffed no one isn't observed.
WITH observed AS (
    SELECT DISTINCT t.event_date AS month_end, t.cost_center
    FROM {{stock}}
    WHERE t.trans_type = 'ACTIVE'
      AND t.event_date > $start
      AND t.event_date <= $end
      AND {{stock_population}}
),
opened AS (
    SELECT t.req_id, last_day(t.opened_date) AS month_end, t.cost_center
    FROM {{source}}
    WHERE t.req_type = 'growth' AND {{population}}
    QUALIFY row_number() OVER (PARTITION BY t.req_id ORDER BY t.interval_number) = 1
)
SELECT o.month_end, o.cost_center, count(g.req_id) AS opened
FROM observed o
LEFT JOIN opened g ON g.month_end = o.month_end AND g.cost_center = o.cost_center
GROUP BY o.month_end, o.cost_center
ORDER BY o.month_end, o.cost_center
