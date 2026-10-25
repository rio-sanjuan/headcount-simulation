# Fit the guidance forecast

The guidance forecast is the aggregate forecast a guided run realizes: event counts per group
and month (attrition, hires, internal starts, promotions, and requisitions opened and closed),
reconciled so its flows agree with its headcount.

```bash
uv run hcsim guidance fit --world worlds/small.duckdb --out guidance/small
```

The artifact it writes is fitted through the world's last snapshot, with a horizon of 12 months
by default (`--horizon` changes it). `--grouping` chooses the group: a cost center by default, or
any level of the org, a job family, a location, and so on.

Two other commands help study the forecast itself:

- `hcsim guidance realized` writes a closed window's realized totals as an artifact, the guidance
  a `guided_actuals` run reads.
- `hcsim guidance ladder` fits the ladder's arms through a cutoff and scores them on the year
  after.

To align the forecast to a plan, give `fit` the plan with `--target`; the
[alignment](../adr/0004-three-valves.md) section of the design describes the valves a plan can
drive.
