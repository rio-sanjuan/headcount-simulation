# Glossary

The public vocabulary of this repository. Names are generic on purpose (see [sanitization](sanitization.md)), and symbols follow the [notation](design/notation.md).

| Term | Meaning |
|---|---|
| Acceptance check | A test over many trajectories that compares a realized statistic with its expectation, within the `stat_check` bound |
| Alignment method | Which valve a plan target drives: `none`, `supply`, `demand`, `hybrid`, or `rate_calibration` |
| Apply step | The single function that writes simulation state, applying each domain's result in a fixed order |
| Arm | One alternative in a study, such as one model for one domain. Every arm is scored against a null arm. |
| Award timing | Per award type, the lag from the dispatch step to the award month, and any calendar months with no award |
| Backfill | A requisition opened to replace a departure or a transfer out |
| Band | A plan's stated uncertainty in heads at a target date. Zero or absent means exact. |
| Capacity cap | A per-cost-center ceiling on committed strength that vetoes requisition creation |
| Category ledger | The per-step record of how much of a shared guidance quota each domain has used |
| Champion | The arm chosen for a domain, after offline scoring and a check in simulation |
| Committed strength | `S = H + O`, active headcount plus open openings |
| Contingent worker | A worker in the `contingent` account group. Only contingent-inclusive populations count them. |
| Conversion | A change of workforce category. A conversion fill hires a contingent worker into a regular role, and a conversion out leaves the regular workforce. |
| Cost center | The finest organizational unit, and the grain of plan targets, capacity caps, and alignment scoring |
| Domain | One kind of workforce or requisition event, handled by one handler, such as attrition or fills |
| Effective ceiling | `Cap_eff`, the tighter of the plan target and the capacity cap |
| Effective forfeiture rate | The per-cohort rate that discounts net expense, recomputed every month |
| Evaluator | A scorer that reads a study's arms and returns a headline, per-arm metrics, and artifacts |
| Fill channel | A route by which an opening fills: external hire, conversion, transfer, campus, pending start, or boundary fill |
| Fill share target | A planner's target share of fills for one or more fill channels that guidance leaves free |
| Fill tree | The nested fill categories: `fills` over `hires` and `internal_starts`, and `hires` over external and conversion hires |
| Fixture | A synthetic input that stands in for the world's actuals in a test. A result built on one is provisional. |
| Forecast scope | Whether a run computes headcount alone (`hcf`) or headcount and SBC (`hcf+sbc`) |
| Forward year | A 12-month window of planned equity value used in sizing, counted after the in-year window |
| Gate | A threshold on an evaluator metric that marks a run provisional or stops it |
| Grain relation | How a subset relates to an artifact's grouping: `COVERED`, `CROSSING`, or `UNREACHABLE` |
| Grant-date fair value | The value per share an award is booked at, which can differ from the month-end close |
| Growth requisition | A requisition opened for net new headcount rather than a replacement |
| Guidance category | One series a guidance forecast can carry, such as `attrition`, `hires`, or `requisitions_opened` |
| Guidance forecast | A fitted forecast of event counts per group and month, which quota sampling realizes exactly |
| Holder ledger | The RSU tracker's record of every granted tranche by recipient, which supports forfeiture and moves between cost centers |
| Known event | A pre-determined event for a named employee or requisition at a given step, also called an anticipated movement |
| Known-answer case | A test on a toy state whose expected outcome is written down before the test first runs |
| Known-event credit | Known headcount changes inside the lookahead window, subtracted from the plan gap |
| Logit shift | A per-group factor on every row's odds, solved so the expected count matches a target rate |
| Natural floor | `H_floor`, the headcount a cost center reaches with no new supply |
| Null arm | The naive benchmark arm every other arm in a domain is measured against |
| Offline stream | Expense for subsidiaries outside the simulated population, run through the same calculator with no attrition |
| Opening | One fillable slot on a requisition |
| Org hierarchy | The rollup of cost centers through numbered levels, `org_level1` to `org_level{n}` |
| Pending start | An accepted offer with a start date, which fills its requisition deterministically |
| Pin | An input a run mode holds at its realized value: movements, volumes, or quota |
| Plan target | `H*`, the plan's headcount for a cost center at a date |
| Planner scope | Where a planner rule applies: globally, to cost centers, or to org nodes at one level |
| Population | A named set of employees and cost centers that a run, a model, and an artifact must agree on |
| Probability tilt | A planner multiplier on event probabilities for chosen domains, scopes, and months |
| Profile | The YAML file that declares a study's scope, inputs, arms, evaluators, gates, and output |
| Provenance | Whether an input or result comes from the synthetic world's actuals, from a fixture, or from a mix |
| Provisional | A flag on a result whose gate failed or whose inputs include a fixture |
| Quota | `K`, the exact count a guided domain realizes per group and month, capped by the eligible rows |
| Quota sampling | Selecting exactly the quota per group, by Gumbel-top-k on the model's probabilities |
| Range position | Where an employee sits within a compensation range, set from their rating tier |
| Rating tier | A performance rating, `R1` (lowest) to `R5` (highest) |
| Recognized expense | Net expense plus every true-up: the amount the books record |
| Reference arm | An arm run for comparison only. It can use another mode, stays out of rankings, and can't be gated. |
| Replayed award | A recorded award granted exactly as recorded, rather than sized by a strategy |
| Requisition | An approved request to fill one or more positions, held in the buffer with its remaining openings |
| Run mode | The rung of the ladder that sets how much of a run is pinned: `replay`, `guided_actuals`, `guided_forecast`, or `unguided` |
| Scorecard | The versioned table of every metric a study produces, one row per arm, evaluator, and metric |
| Seed manager | The component that derives each random generator from the seed, a subsystem name, and the trajectory index |
| Seed tranches | Equity granted before the run date that's still vesting |
| Specificity | A planner rule's precedence: a cost center outranks an org node, which outranks global |
| Spill | Reassigning a transfer-assigned requisition with no candidate to another free fill channel |
| Study | A declared experiment: a window, a population, a run mode, arms, evaluators, and gates |
| Subset | A declared part of a population that a run simulates on its own |
| Supply rule | The formula that sizes target-seeking growth: `strength`, `expected_fill`, or `soft_bridge` |
| `t*` | The evaluation step: the last simulated step inside the plan's actively planned months |
| Target total compensation | The annual compensation a sizing strategy aims for |
| Tech indicator | An attribute that marks a role as software engineering, other technical, or non-technical |
| Termination program | Planned terminations for named employees over a window, entered as known events |
| Trajectory | One independent Monte Carlo replicate of a run |
| Tranche | Shares granted in one award month that vest in one vest month, for one combination of attributes |
| Transaction code | A typed code on the workforce fact, such as `HIRE`, `TERM`, `XFER_IN`, or `XFER_OUT` |
| True-up | An expense adjustment for attrition, for a change in the effective forfeiture rate, or for the shares that actually vest |
| Ungranted grant | In replay, a grant whose window lies within history's record of awards and finds none, so it's granted nothing, as history granted none |
| Valve | One of three separate controls: supply from the plan, flow from guidance, and the capacity veto |
| Vest template | The monthly vesting fractions per forward year for an award type |
| WAPE | Weighted absolute percentage error: absolute error summed over cells, divided by summed actuals |
| Window closure | Whether a study's months have actuals to score: wholly open, partially closed, or wholly closed |
