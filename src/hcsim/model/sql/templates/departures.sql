-- Departures: every termination and transfer out of the population in ($start, $end], with
-- the person's role, labelled by whether a backfill requisition was opened for it. A backfill
-- names the person it replaces and belongs to their departure nearest it in time, the earlier
-- on a tie, so someone who leaves twice has each backfill counted once.
WITH departures AS (
    SELECT t.employee_id, t.event_date, t.trans_type AS departure, t.cost_center,
           j.job_family, j.job_level, j.tech_indicator, t.location, t.company, t.is_manager
    FROM {{source}}
    WHERE t.trans_type IN ('TERM', 'XFER_OUT') AND {{population}}
),
backfills AS (
    SELECT req_id, any_value(backfill_of) AS employee_id, min(opened_date) AS opened_date
    FROM req_interval
    WHERE backfill_of IS NOT NULL
    GROUP BY req_id
),
matched AS (
    SELECT DISTINCT employee_id, event_date
    FROM (
        SELECT b.employee_id, d.event_date
        FROM backfills b
        JOIN departures d ON d.employee_id = b.employee_id
        QUALIFY row_number() OVER (
            PARTITION BY b.req_id
            ORDER BY abs(datediff('day', d.event_date, b.opened_date)), d.event_date
        ) = 1
    )
)
SELECT d.*, (m.employee_id IS NOT NULL) AS label
FROM departures d
LEFT JOIN matched m ON m.employee_id = d.employee_id AND m.event_date = d.event_date
WHERE d.event_date > $start AND d.event_date <= $end
ORDER BY d.event_date, d.employee_id, d.departure
