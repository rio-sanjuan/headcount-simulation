-- Ratings: every rating in ($start, $end] of someone of the population, with their role at
-- their last month-end in the stock before the cycle and the rating before it (no scale and
-- rank 0 for a first rating), so consecutive pairs train transitions and first ratings train
-- entries.
WITH stock AS (
    SELECT t.employee_id, t.event_date AS month_end, t.cost_center, j.job_family, j.job_level,
           j.tech_indicator, t.location, t.company, t.is_manager
    FROM {{source}}
    WHERE t.trans_type = 'ACTIVE'
      AND t.event_date >= $start - INTERVAL 1 MONTH
      AND t.event_date <= $end
      AND {{population}}
),
ordered AS (
    SELECT employee_id, cycle_date, rating_scale, rating_rank, is_first,
           lag(rating_scale) OVER w AS previous_scale,
           coalesce(lag(rating_rank) OVER w, 0) AS previous_rank
    FROM rating
    WINDOW w AS (PARTITION BY employee_id ORDER BY cycle_date)
)
SELECT r.employee_id, r.cycle_date, s.cost_center, s.job_family, s.job_level, s.tech_indicator,
       s.location, s.company, s.is_manager, r.previous_scale, r.previous_rank, r.rating_scale,
       r.rating_rank, r.is_first
FROM ordered r
ASOF JOIN stock s ON s.employee_id = r.employee_id AND r.cycle_date > s.month_end
WHERE r.cycle_date > $start AND r.cycle_date <= $end
ORDER BY r.cycle_date, r.employee_id
