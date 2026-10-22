-- Requisition roles: every growth requisition of the population opened in ($start, $end], with
-- the role it was opened for and the org it opened in, from its cost center up.
SELECT t.req_id, t.opened_date, t.cost_center, COLUMNS('^org_level\d+$'), j.job_level,
       t.location, j.job_family, j.tech_indicator, t.is_manager, t.company
FROM {{source}}
WHERE t.req_type = 'growth'
  AND t.opened_date > $start
  AND t.opened_date <= $end
  AND {{population}}
QUALIFY row_number() OVER (PARTITION BY t.req_id ORDER BY t.interval_number) = 1
ORDER BY t.opened_date, t.req_id
