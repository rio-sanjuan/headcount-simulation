-- Seed: every employee of the population in the stock at a month-end ($snapshot).
SELECT t.employee_id, t.cost_center, j.job_family, j.job_level, j.tech_indicator,
       t.location, t.company, t.is_manager, t.hire_date
FROM {{source}}
WHERE t.trans_type = 'ACTIVE'
  AND t.event_date = $snapshot
  AND {{population}}
ORDER BY t.employee_id
