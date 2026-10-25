# Domains

A run moves its workforce and its open requisitions forward a month at a time. Each month, the
domains run in a fixed order, and each one's change is applied before the next reads the state,
so a later domain always sees what the earlier ones left. A domain is pure: it reads the state
and returns a declarative change, and the apply step is the only writer.

| Step | Domain | What it does |
|---|---|---|
| 1 | Replayed requisitions | In a replay, history's requisitions open first. |
| 2–4 | [Known events](known-events.md) | Conversions out, reorganizations, and other movements decided before the run. |
| 5 | [Performance](performance.md) | Ratings change in the cycle month. |
| 6 | [Promotion](promotion.md) | Who moves up a level. |
| 7 | [Attrition](attrition.md) | Who leaves. |
| 8 | [Backfill](backfill.md) | Requisitions opened to replace departures. |
| 9–10 | [Awards](awards.md) | The annual award and anticipated awards. |
| 11 | [Pending starts](pending-starts.md) | Accepted offers that start this month. |
| 12 | [Fills](fills.md) | Which open requisitions fill, and by which channel. |
| 13 | [Transfers](transfers.md) | Who takes each opening assigned to an internal transfer. |
| 14 | [Campus](campus.md) | Campus requisitions on their schedule. |
| 15 | [Cancellation](cancellation.md) | Requisitions that leave unfilled. |
| 16 | Backfill for transfers | Requisitions opened to replace transfers out. |
| 17 | [Growth](growth.md) | New requisitions, at the model's rate, from guidance, or toward a plan. |

Where a guidance forecast covers a group, a domain realizes the group's quota exactly (quota
sampling, [ADR 0002](../adr/0002-quota-sampling.md)); elsewhere, every row draws on its own.

Each page below is the domain's own documentation from the code, and a **data findings**
section: what fitting and running the domain on the synthetic world showed.

::: hcsim.engine.loop
