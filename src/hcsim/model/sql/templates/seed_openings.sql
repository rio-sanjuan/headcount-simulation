-- Seed: every requisition of the population open at a month-end ($snapshot), with the
-- openings it still has to fill then.
SELECT t.req_id, t.req_type, t.cost_center, j.job_family, j.job_level, j.tech_indicator,
       t.location, t.company, t.is_manager, t.openings, t.opened_date
FROM {{source}}
WHERE t.interval_start <= $snapshot
  AND (t.closed_date IS NULL OR t.closed_date > $snapshot)
  AND {{population}}
ORDER BY t.req_id
