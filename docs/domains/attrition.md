# Attrition

::: hcsim.engine.domains.attrition

## Data findings

- The fitted hazard recovers the world's: 0.00935 a month against the truth's 0.00920 on the
  small world, each level within about 10%.
- A boosted model beats the null arm on the medium world (log loss 0.04624 to 0.04514); on the
  small world, a logistic model wins.
- Under guidance, attrition realizes its quota exactly. Over a closed window, quota sampling's
  monthly departures match history in every cost center, while drawing each employee
  independently at the same rates misses by 30% cell by cell (M8's `why_quota_sampling`).
- In an unguided run, realized attrition stays within the statistical bound of the fitted rate.
