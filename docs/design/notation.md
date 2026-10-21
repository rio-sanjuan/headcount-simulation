# Notation

One symbol set covers stocks, rates, flows, quotas, and metrics. Code, docs, and reports use these symbols, and a new quantity extends this page rather than inventing a parallel name.

## Conventions

- **A letter names a role, and a subscript names a channel.** `F` is an inflow through the hiring pipeline, `V` is a non-attrition outflow, and `B` is a backfill requisition. The channels are `att` (attrition), `ext` (external), `conv` (conversion), `camp` (campus), `tr` (transfer), `pend` (pending start), and `bdy` (boundary).
- **Every quantity carries its indices.** Write `H(c, t)`, never `H_c`. The step index tells a per-step target from a terminal one.
- **A star marks a plan quantity, and a hat marks a fitted one.** `H(c, t)` is realized headcount, `H*(c, t)` is the plan target, and `Ĥ(c, t)` is the fitted headcount series in a guidance forecast.
- **Attrition keeps its own letter.** The outflows are `{A, V_tr, V_conv}`. Read `V` as a non-attrition outflow, never as all outflows.

## Indices

| Symbol | Meaning |
|---|---|
| `c` | Cost center: the plan grain, and the grain alignment is scored at |
| `g` | Guidance group: the grouping of one guidance series, coarser or finer than `c` |
| `d` | A domain output that consumes a guidance category |
| `t` | Step: one simulated month, with `t = 0` the first |
| `T` | Target period: the step by which the plan must be met |
| `t*` | Evaluation step, chosen inside the plan's actively planned months |
| `N` | Horizon length in months |
| `j`, `J` | Trajectory index and count |

A **trajectory** is one Monte Carlo replicate, and a **path** is one time series. Every claim about variance is about trajectories.

## Stocks

| Symbol | Meaning |
|---|---|
| `H(c, t; j)` | Realized active headcount |
| `H*(c, t)` | Plan target headcount |
| `Ĥ(g, t)` | Fitted headcount, the anchor of the forecast-coherency identity |
| `O(c, t; j)` | Open openings: remaining fillable openings summed over open requisitions |
| `O_req(c, t; j)` | Open requisition rows, a diagnostic only |
| `S(c, t; j)` | Committed strength, `H + O` |
| `Cap(c)` | Capacity cap |
| `Cap_eff(c, t)` | Effective ceiling: the tighter of `H*(c, t)` and `Cap(c)`, where either exists |
| `H_floor(c, t)` | Natural floor: the seed surviving attrition, plus expected fills of openings already open, plus known net inflows |
| `band(c, d)` | The plan's uncertainty in heads at target date `d`. Zero or absent means exact. |

`O` counts openings, not rows. A requisition carries one or more fillable openings, and only openings turn into headcount, so `H* − S` is a coherent subtraction and `H* − H − O_req` isn't.

## Rates

Rates come in two classes, told apart by their denominator.

| Symbol | Rate | Denominator |
|---|---|---|
| `a(c, t)` | Attrition hazard | Active heads |
| `f(c, t)` | Fill hazard | Open openings |
| `b_att(c)` | Backfill per departure | Departures |
| `b_tr(c)` | Backfill per transfer out | Transfers out |
| `u(c)` | Requisitions that exit unfilled, per requisition opened | Requisitions opened, a flow |
| `y(c)` | Eventual fill yield, per open opening | Open openings, a stock |

`b_att(c) · (1 − u(c))` is the effective replacement rate. `y` and `1 − u` aren't interchangeable: the stock of open openings is older on average than the flow of new requisitions, so it's less likely to fill. `P(fill by ℓ)` is the probability that an opening fills within `ℓ` steps, which tends to `y`.

## Flows

Flows that change headcount:

| Symbol | Flow | Consumes an opening |
|---|---|---|
| `A` | Departures (attrition) | — |
| `F_ext` | External fill | Yes |
| `F_conv` | Conversion fill | Yes |
| `F_camp` | Campus fill | Yes |
| `F_tr` | Transfer fill | Yes |
| `F_pend` | Pending starts | Yes |
| `F_bdy` | Boundary fill, injected in replay only | No |
| `V_tr` | Transfers out | — |
| `V_conv` | Conversions out | — |
| `R⁺`, `R⁻` | Reorganizations in and out | No |
| `W⁺`, `W⁻` | Other movements in and out | No |

Flows that change requisitions: `G`, growth requisitions opened; `B_att`, backfill opened on a departure; `B_tr`, backfill opened on a transfer out; and `U`, requisitions that exit unfilled. `G`, `B_att`, and `B_tr` draw on one shared requisitions-opened quota. `Π` denotes promotions, which change an attribute but no stock.

`U` counts every requisition that leaves the open population without a fill, including approved requisitions that simply end, which is why it isn't called "cancellations".

## Quotas and lag

A quota `K(g, d, t)` is the exact count quota sampling realizes for domain output `d` in group `g` at step `t`. `K_A` and `K_F` name the attrition and fill quotas. A quota is an input and a flow symbol is the realized count. They coincide only when eligibility doesn't bind, because a domain realizes `min(K, eligible)`.

`L(c)` is the time from opening to fill. It emerges from the simulation rather than being set. `L̂` is the value a rule reads, the credit lookahead. `β` is the supply gain, the share of a gap a supply rule closes per step, and `k(c, t)` is the soft-bridge gain toward the next target date.

## The two identities

They're different objects.

The **forecast-coherency identity** is what the guidance pipeline enforces offline, per group and month, within a tolerance. Reconciliation holds the anchor `Ĥ` fixed and adjusts the flows.

```
Ĥ(g, t) − Ĥ(g, t−1) = F_ext + F_conv + F_tr − A − V_conv
```

The **runtime accounting identity** is what simulation state obeys, and it carries every flow.

```
H(c, t+1; j) = H(c, t; j) + F − A − V_tr − V_conv + R⁺ − R⁻ + W⁺ − W⁻
O(c, t+1; j) = O(c, t; j) + G + B_att + B_tr − U − (F_ext + F_conv + F_camp + F_tr + F_pend)
```

Here `F` is the sum of the six fill channels. The difference between the two identities is exactly the set of flows the guidance pipeline can't forecast.

### A worked example

One synthetic cost center `c` in one month. Every requisition carries one opening. The month starts with `H = 120` and `O = 8`, so `S = 128`.

- Three employees leave (`A = 3`), and backfill opens two requisitions (`B_att = 2`).
- Three external hires, one transfer in, and one pending start each fill an opening (`F_ext = 3`, `F_tr = 1`, `F_pend = 1`).
- One requisition exits unfilled (`U = 1`).

```
H(c, t+1) = 120 + (3 + 1 + 1) − 3     = 122
O(c, t+1) = 8 + 2 − 1 − (3 + 1 + 1)   = 4
S(c, t+1) = 122 + 4                   = 126
```

If `c` is its own guidance group, the flows the coherency identity carries sum to `F_ext + F_tr − A = 1`, while headcount rose by 2. The gap is the pending start, a flow with no series.

Committed strength fell by 2. A fill that consumes an opening moves one unit from `O` to `H` and leaves `S` unchanged, so `S` moves only through new openings (+2), unfilled exits (−1), and flows that consume no opening, here the departures (−3).

## Metrics

Over trajectories `j = 1..J` at the evaluation step `t*`:

```
bias(c)   = | mean_j H(c, t*; j) − H*(c, t*) |
spread(c) = sd_j H(c, t*; j)
D(c)      = max_j | H(c, t*; j) − H*(c, t*) |
```

Bias measures distance from plan, and calibration measures whether the spread honestly represents uncertainty. Scores are taken at cost-center grain, never at the total, because plan versions can agree on totals while disagreeing on most cost centers.
