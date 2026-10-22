-- Training extract: one row per employee of the population in the stock at each month-end in
-- [$start, $end), labelled by whether one of the events happens to them in the next month.
WITH at_risk AS (
    SELECT t.employee_id, t.event_date AS month_end{{attributes?}}
    FROM {{source}}
    WHERE t.trans_type = 'ACTIVE'
      AND t.event_date >= $start
      AND t.event_date < $end
      AND {{population}}
),
events AS (
    SELECT DISTINCT employee_id, last_day(event_date) AS month_end
    FROM workforce_txn
    WHERE trans_type IN ({{codes}})
)
SELECT r.*, (e.employee_id IS NOT NULL) AS label
FROM at_risk r
LEFT JOIN events e
  ON e.employee_id = r.employee_id
 AND e.month_end = last_day(r.month_end + INTERVAL 1 MONTH)
ORDER BY r.month_end, r.employee_id
