-- Realized flows: a measure's signed count per month and group, for events of the population
-- in ($start, $end].
SELECT date_trunc('month', t.event_date) AS month, {{group}} AS "group",
       sum({{expression}}) AS value
FROM {{source}}
WHERE t.trans_type IN ({{codes}})
  AND t.event_date > $start
  AND t.event_date <= $end
  AND {{population}}
GROUP BY ALL
ORDER BY ALL
