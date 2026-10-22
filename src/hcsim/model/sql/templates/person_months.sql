-- Person-months at risk: one row per employee of the population in the stock at each
-- month-end in [$start, $end), with their tenure in months, their latest rating (rank 0 and no
-- scale if unrated), and the calendar month ahead, labelled by whether the events net to at
-- least one for them the next month.
WITH at_risk AS (
    SELECT t.employee_id, t.event_date AS month_end{{attributes?}},
           datediff('month', t.hire_date, t.event_date) AS tenure_months,
           month(t.event_date + INTERVAL 1 MONTH) AS next_month
    FROM {{source}}
    WHERE t.trans_type = 'ACTIVE'
      AND t.event_date >= $start
      AND t.event_date < $end
      AND {{population}}
),
rated AS (
    SELECT r.*, coalesce(g.rating_rank, 0) AS rating_rank, g.rating_scale
    FROM at_risk r
    ASOF LEFT JOIN rating g
      ON g.employee_id = r.employee_id AND r.month_end >= g.cycle_date
),
events AS (
    SELECT employee_id, last_day(event_date) AS month_end
    FROM workforce_txn
    WHERE trans_type IN ({{codes}})
    GROUP BY ALL
    HAVING sum({{net}}) > 0
)
SELECT r.*, (e.employee_id IS NOT NULL) AS label
FROM rated r
LEFT JOIN events e
  ON e.employee_id = r.employee_id
 AND e.month_end = last_day(r.month_end + INTERVAL 1 MONTH)
ORDER BY r.month_end, r.employee_id
