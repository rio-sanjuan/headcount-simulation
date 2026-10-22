-- Requisition-months: one row per requisition of the population and month in the window it's
-- open in, from the month it was approved, so a requisition filled in its first month still
-- has a row. A row is keyed by its month-end, in ($start, $end]. Features are as at the
-- month's start, or at approval in the approval month: the requisition's age in months and the
-- openings it has left. Outcomes are the month's: how many openings filled, the channel of the
-- earliest fill, and whether the requisition exited unfilled.
WITH reqs AS (
    SELECT t.req_id, t.req_type, t.cost_center, j.job_family, j.job_level, j.tech_indicator,
           t.location, t.company, t.is_manager, t.opened_date
    FROM {{source}}
    WHERE {{population}}
    QUALIFY row_number() OVER (PARTITION BY t.req_id ORDER BY t.interval_number) = 1
),
intervals AS (
    SELECT req_id, interval_number, openings, closed_date, close_reason,
           last_day(closed_date) AS closing_month,
           starts_with(close_reason, 'filled') AS filled
    FROM req_interval
    WHERE req_id IN (SELECT req_id FROM reqs)
),
spans AS (
    SELECT req_id, max(closed_date) AS last_closed, bool_or(closed_date IS NULL) AS still_open
    FROM intervals
    GROUP BY req_id
),
grid AS (
    SELECT r.*, c.month_end,
           greatest(r.opened_date, CAST(date_trunc('month', c.month_end) AS DATE)) AS as_of
    FROM reqs r
    JOIN spans s ON s.req_id = r.req_id
    JOIN calendar c
      ON c.month_end >= last_day(r.opened_date)
     AND (s.still_open OR c.month_end <= last_day(s.last_closed))
    WHERE c.month_end > $start AND c.month_end <= $end
),
state AS (
    SELECT g.req_id, g.month_end,
           arg_min(i.openings, i.interval_number)
               FILTER (i.closed_date IS NULL OR i.closed_date >= g.as_of) AS openings,
           count(*) FILTER (i.filled AND i.closing_month = g.month_end) AS fills,
           replace(
               arg_min(i.close_reason, i.interval_number)
                   FILTER (i.filled AND i.closing_month = g.month_end),
               'filled_', ''
           ) AS fill_channel,
           coalesce(bool_or(i.close_reason = 'unfilled' AND i.closing_month = g.month_end), false)
               AS closed_unfilled
    FROM grid g
    JOIN intervals i ON i.req_id = g.req_id
    GROUP BY g.req_id, g.month_end
)
SELECT g.req_id, g.month_end, g.req_type, g.cost_center, g.job_family, g.job_level,
       g.tech_indicator, g.location, g.company, g.is_manager,
       datediff('month', g.opened_date, g.as_of) AS age_months, s.openings, s.fills,
       s.fill_channel, s.closed_unfilled
FROM grid g
JOIN state s ON s.req_id = g.req_id AND s.month_end = g.month_end
ORDER BY g.month_end, g.req_id
