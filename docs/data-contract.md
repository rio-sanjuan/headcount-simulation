# Data contract

This page specifies the tables the synthetic world writes. hcsim's readers depend on them: seeding, the guidance ingest, the study inputs, and the coherence checks. Any other source of the same tables should follow this contract too. The page lists each table's grain and every column, then the transaction codes, the truth schema, the quirks, and how the contract changes.

**Contract version: 1.0.** The world records the version it follows in its manifest, as `contract_version`.

## Conventions

- **Warehouse.** A DuckDB file has two schemas. `main` holds the source tables and one view. `truth` holds what only a simulator knows (see [The truth schema](#the-truth-schema)). Readers of source data never read `truth`, except to record which world a run used. Tests and recovery reports read the rest.
- **Types.** Dates are `DATE`, with no times. Labels are `VARCHAR`, counts and identifiers are `BIGINT`, money and rates are `DOUBLE`, and flags are `BOOLEAN`.
- **Labels are opaque.** Labels are invented codes, such as `CC-01`, `ORG-A`, `LOC-1`, `CO-1`, and `ACCT-100`. Readers must not parse meaning out of them. Meaning comes from the dimension tables.
- **Identifiers.** Employee IDs start at 100001, requisition IDs at 500001, and award IDs at 900001. An ID is never reused.
- **Month-ends.** Monthly facts are dated on the month's last day. `calendar` lists every month-end in the world's window.
- **Event and reporting dates.** A transaction carries the date its event took effect and the date it reached the warehouse. Reading by event date reconstructs what happened. Reading by reporting date shows what was known at a time.
- **Regular headcount.** Headcount means employees on accounts whose `account_group` is `regular`. Contingent and temporary workers book to their own accounts, and readers leave them out of headcount measures.
- **Missing values.** A column may be `NULL` only where its description says so.
- **Row order.** Row order isn't part of the contract, except that `interval_number` orders a requisition's intervals.
- **The cost-center column.** The leaf cost center may go by another name in some tables (see [`contract_columns`](#contract_columns)). Readers should resolve it from that table, not assume `cost_center`.

## Source tables

### `calendar`

One row per month-end in the world's window.

| column | type | description |
|---|---|---|
| `month_end` | `DATE` | A month-end, from the first month to the last. |

### `contract_columns`

One row per table that holds a leaf cost center, and the column that holds it.

| column | type | description |
|---|---|---|
| `table_name` | `VARCHAR` | A table in `main`. |
| `role` | `VARCHAR` | What the column holds. Always `cost_center` in this version. |
| `column_name` | `VARCHAR` | The column that holds it in that table. |

### `dim_account`

One row per account.

| column | type | description |
|---|---|---|
| `account` | `VARCHAR` | The account label. |
| `account_group` | `VARCHAR` | `regular`, `contingent`, or `temporary`. |

### `dim_company`

One row per legal company.

| column | type | description |
|---|---|---|
| `company` | `VARCHAR` | The company label. |
| `planning_company` | `VARCHAR` | The company it rolls up to for planning. |
| `is_primary` | `BOOLEAN` | Whether it's the primary company. Exactly one is. |
| `is_offline` | `BOOLEAN` | Whether it's an offline subsidiary. Its equity is valued by appraisal, with no market price. |

### `dim_job`

One row per job: a family, a level, and whether the role manages people.

| column | type | description |
|---|---|---|
| `job_code` | `VARCHAR` | The job label. |
| `job_title` | `VARCHAR` | A title. A manager job's title names it as one. |
| `job_family` | `VARCHAR` | The job family. |
| `job_level` | `BIGINT` | The level, from 1 up. Higher is more senior. |
| `level_name` | `VARCHAR` | The level's name. |
| `is_executive` | `BOOLEAN` | Whether the level is in the executive band. |
| `manager_title` | `BOOLEAN` | Whether the job manages people. In the presets, managers start at level 4. |
| `tech_indicator` | `VARCHAR` | The family's technical group: `software_engineering`, `other_technical`, or `non_technical`. |

### `dim_location`

One row per work location.

| column | type | description |
|---|---|---|
| `location` | `VARCHAR` | The location label. |
| `country` | `VARCHAR` | The country it's in. |
| `region` | `VARCHAR` | The region the country is in. |

### `dim_org`

One row per leaf cost center, with its path up the org hierarchy.

| column | type | description |
|---|---|---|
| `cost_center` | `VARCHAR` | The leaf cost center. |
| `org_level<k>` | `VARCHAR` | The cost center's ancestor at level *k*, from `org_level1` (the top) to the deepest level. The hierarchy's depth varies. A branch that ends early repeats its leaf at every deeper level. |
| `cost_type` | `VARCHAR` | `fixed` or `variable`. |
| `field_type` | `VARCHAR` | `corporate` or `field`, by the cost center's top-level org. |

### `dim_rating`

One row per rating label on each scale.

| column | type | description |
|---|---|---|
| `rating` | `VARCHAR` | The rating label. |
| `rating_rank` | `BIGINT` | Its rank on its scale, from 1, the lowest. |
| `rating_scale` | `VARCHAR` | `original`, or `revised` for a scale introduced later. |

### `dim_trans_type`

One row per transaction code. The codes are listed in [Transaction codes](#transaction-codes).

| column | type | description |
|---|---|---|
| `trans_type` | `VARCHAR` | The code. |
| `disposition` | `VARCHAR` | `measured`, `excluded`, or `ignored`. |
| `amount` | `BIGINT` | The code's signed effect on headcount: 1, 0, or -1. |
| `description` | `VARCHAR` | What the code records. |

### `dim_vest_schedule`

One row per tranche of each vest schedule.

| column | type | description |
|---|---|---|
| `schedule` | `VARCHAR` | The schedule: an award type's own, or `executive`. |
| `vest_number` | `BIGINT` | The tranche, from 1. |
| `months_after_award` | `BIGINT` | When it vests, in months after the award date. |
| `share` | `DOUBLE` | Its share of the award. A schedule's shares sum to 1. |

### `workforce_txn`

One row per transaction, plus one `ACTIVE` row per person in post at each month-end. The `_previous` columns hold the attribute as it was just before the row's event. For a row that changes nothing, they equal the current values.

| column | type | description |
|---|---|---|
| `employee_id` | `BIGINT` | The person. |
| `reporting_date` | `DATE` | When the row reached the warehouse. Stocks report on their month-end. A notice reports on the day it's given. Other events report on the day, or a few days later. |
| `event_date` | `DATE` | When the event took effect. For `ACTIVE`, the month-end. |
| `trans_type` | `VARCHAR` | The code, from `dim_trans_type`. |
| `amount` | `BIGINT` | The row's signed effect on its account's headcount. |
| `cost_center` | `VARCHAR` | The leaf cost center after the event. |
| `cost_center_previous` | `VARCHAR` | The leaf cost center before it. |
| `job_code` | `VARCHAR` | The job after the event. |
| `job_code_previous` | `VARCHAR` | The job before it. |
| `location` | `VARCHAR` | The location after the event. |
| `location_previous` | `VARCHAR` | The location before it. |
| `company` | `VARCHAR` | The company after the event. |
| `company_previous` | `VARCHAR` | The company before it. |
| `account` | `VARCHAR` | The account after the event. |
| `account_previous` | `VARCHAR` | The account before it. |
| `is_manager` | `BOOLEAN` | Whether the role manages people, after the event. |
| `is_manager_previous` | `BOOLEAN` | Whether it did before. |
| `hire_date` | `DATE` | The person's hire date. |

Within each account and cost center, a month-end's `ACTIVE` count is the previous month-end's count plus the month's summed `amount`, by event date. Every quirk keeps that identity.

### `workforce_agg`

The transactions pre-summed. There's one row per month, cost center, job family, level, location, company, account, and code. The guidance pipeline reads this table rather than the transactions.

| column | type | description |
|---|---|---|
| `period` | `DATE` | The month-end of the events' month. |
| `cost_center` | `VARCHAR` | The leaf cost center. |
| `job_family` | `VARCHAR` | The job family. |
| `job_level` | `BIGINT` | The level. |
| `location` | `VARCHAR` | The location. |
| `company` | `VARCHAR` | The company. |
| `account` | `VARCHAR` | The account. |
| `trans_type` | `VARCHAR` | The code. |
| `headcount` | `BIGINT` | The number of transactions summed. For `ACTIVE`, the month-end stock. |
| `amount` | `BIGINT` | Their summed `amount`. |

Apart from rows with an ignored code that only the aggregate holds, it equals `workforce_txn` grouped by its keys.

### `rating`

One row per person and calibration cycle.

| column | type | description |
|---|---|---|
| `employee_id` | `BIGINT` | The person. |
| `cycle_date` | `DATE` | The cycle's effective date, the first day of a cycle month. Cycles before the window rate the starting workforce. |
| `rating` | `VARCHAR` | The rating, on the scale reported at the time. |
| `rating_rank` | `BIGINT` | Its rank on that scale, from 1, the lowest. |
| `rating_scale` | `VARCHAR` | `original` or `revised`. |
| `is_first` | `BOOLEAN` | Whether it's the person's first rating. |

### `req_interval`

One row per requisition and state interval. A requisition's state changes when an opening fills, when a no-show gives an opening back, or when it closes.

| column | type | description |
|---|---|---|
| `req_id` | `BIGINT` | The requisition. |
| `interval_number` | `BIGINT` | The interval's place in the requisition's history, from 1. Several intervals can share a day. |
| `req_type` | `VARCHAR` | `growth`, `backfill`, or `campus`. |
| `cost_center` | `VARCHAR` | The leaf cost center. |
| `job_code` | `VARCHAR` | The job. |
| `job_family` | `VARCHAR` | The job's family. |
| `job_level` | `BIGINT` | The job's level. |
| `location` | `VARCHAR` | The location. |
| `company` | `VARCHAR` | The company. |
| `is_manager` | `BOOLEAN` | Whether the role manages people. |
| `openings` | `BIGINT` | The openings still to fill during the interval. The first interval's openings are the ones the requisition opened with. |
| `opened_date` | `DATE` | When the requisition opened. |
| `interval_start` | `DATE` | When the interval began. |
| `closed_date` | `DATE` | When the interval ended. `NULL` while it's the current interval of a requisition still open. |
| `close_reason` | `VARCHAR` | How it ended: `filled_external`, `filled_transfer`, `filled_conversion`, `filled_campus`, `reopened` (a no-show gave an opening back), or `unfilled` (the requisition closed with its openings). `NULL` when `closed_date` is. |
| `filled_by` | `BIGINT` | The person who filled the opening, for a fill. Otherwise `NULL`. |
| `backfill_of` | `BIGINT` | For a backfill, the person whose departure it replaces. Otherwise `NULL`. |

A fill takes one opening, and the next interval starts the same day. A no-show ends an open interval as `reopened`, and the next interval has one opening more. If the requisition had already filled its last opening, a no-show opens a new interval with one opening. Only the last interval may be open or end `unfilled`. Requisitions still open at the world's end have no closing date: their closures are right-censored.

### `req_snapshot` (view)

One row per requisition open at each month-end, as a month-end snapshot feed would record it. A requisition that opens and closes within one month never appears.

| column | type | description |
|---|---|---|
| `snapshot_date` | `DATE` | The month-end. |
| `req_id` | `BIGINT` | The requisition. |
| `req_type` | `VARCHAR` | `growth`, `backfill`, or `campus`. |
| `cost_center` | `VARCHAR` | The leaf cost center. The view always uses this name. |
| `job_code` | `VARCHAR` | The job. |
| `location` | `VARCHAR` | The location. |
| `company` | `VARCHAR` | The company. |
| `is_manager` | `BOOLEAN` | Whether the role manages people. |
| `openings` | `BIGINT` | The openings still to fill at the month-end. |
| `state` | `VARCHAR` | The feed's state label, always meaning open. Its spelling can change partway through (see [Quirks](#quirks)). |
| `opened_date` | `DATE` | When the requisition opened. |

### `pending_start`

One row per accepted offer and month-end between the offer and the start.

| column | type | description |
|---|---|---|
| `snapshot_date` | `DATE` | The month-end. |
| `employee_id` | `BIGINT` | The person, whose ID is reserved at the offer. |
| `req_id` | `BIGINT` | The requisition the offer filled. |
| `cost_center` | `VARCHAR` | The leaf cost center. |
| `job_code` | `VARCHAR` | The job. |
| `location` | `VARCHAR` | The location. |
| `company` | `VARCHAR` | The company. |
| `offer_date` | `DATE` | When the offer was accepted, the day the opening closed. |
| `start_date` | `DATE` | When the person is due to start. Some never do. |

Each row satisfies `offer_date <= snapshot_date < start_date`.

### `plan_headcount`

Published headcount plans. There's one row per plan vintage, cost center, job family, and month, plus a hedge row per top-level org, vintage, and month. Cells with nothing planned are left out.

| column | type | description |
|---|---|---|
| `vintage` | `VARCHAR` | The vintage, named by its publication month. |
| `published_date` | `DATE` | When it was published. |
| `cost_center` | `VARCHAR` | The leaf cost center, or a hedge's pseudo cost center, which belongs to no org. |
| `job_family` | `VARCHAR` | The job family. `NULL` on hedge rows. |
| `month_end` | `DATE` | The month planned for. A vintage runs past the world's end. |
| `is_hedge` | `BOOLEAN` | Whether the row is a hedge. |
| `target_headcount` | `DOUBLE` | The target. Targets are fractional, and a few cells are negative, but no cost center's monthly total is. |

### `comp_range`

One row per job family, level, and location, with every combination present, for award sizing.

| column | type | description |
|---|---|---|
| `job_family` | `VARCHAR` | The job family. |
| `job_level` | `BIGINT` | The level, executive levels included. |
| `location` | `VARCHAR` | The location. |
| `min_base` | `BIGINT` | The bottom of the base pay range. |
| `max_base` | `BIGINT` | The top of the base pay range. |
| `min_target` | `BIGINT` | The bottom of the target total compensation range: base plus the level's equity. |
| `max_target` | `BIGINT` | The top of that range. |
| `base_adj` | `DOUBLE` | A multiplier on the base figures for adjustments since the ranges were set. It's the same on every row. |

### `stock_price`

One row per month-end.

| column | type | description |
|---|---|---|
| `month_end` | `DATE` | The month-end. |
| `close` | `DOUBLE` | The share price at that month's last close. |

### `stock_price_daily`

One row per trading day.

| column | type | description |
|---|---|---|
| `price_date` | `DATE` | The trading day. |
| `close` | `DOUBLE` | The closing price. |
| `fair_value` | `DOUBLE` | The grant-date fair value per share: a trailing average of closes, so it differs from the close. |

### `equity_award`

One row per award.

| column | type | description |
|---|---|---|
| `award_id` | `BIGINT` | The award. |
| `employee_id` | `BIGINT` | Its holder at grant. |
| `award_type` | `VARCHAR` | `new_hire`, `annual`, `promotion`, `transfer`, `retention`, or `off_cycle`. |
| `award_date` | `DATE` | The grant date, a business day. |
| `shares` | `BIGINT` | Shares granted, at least 1. |
| `fair_value` | `DOUBLE` | The grant value per share. For the primary plan, that's the day's fair value. For an offline plan, it's the latest appraisal. |
| `cost_center` | `VARCHAR` | The holder's leaf cost center at grant. |
| `company` | `VARCHAR` | The holder's company at grant. |
| `job_level` | `BIGINT` | The holder's level at grant. |
| `plan` | `VARCHAR` | `primary`, or `offline` for an offline subsidiary's plan. |
| `schedule` | `VARCHAR` | The vest schedule, from `dim_vest_schedule`. |

### `equity_vest`

One row per tranche.

| column | type | description |
|---|---|---|
| `award_id` | `BIGINT` | The award. |
| `vest_number` | `BIGINT` | The tranche, from 1. |
| `vest_date` | `DATE` | When it vests. Tranches can vest after the world's end. |
| `shares` | `BIGINT` | Its shares. An award's tranches sum to its shares, with the last absorbing rounding. |
| `status` | `VARCHAR` | `vested`, `forfeited` (the holder left first), or `unvested` (still to vest at the world's end). |
| `forfeit_date` | `DATE` | When a forfeited tranche was lost, the day the holder left. Otherwise `NULL`. |

### `equity_valuation`

One row per award valuation: its grant, and every quarter-end appraisal of an offline plan's award while it's held.

| column | type | description |
|---|---|---|
| `award_id` | `BIGINT` | The award. |
| `valuation_date` | `DATE` | When it was valued. |
| `value_per_share` | `DOUBLE` | The value per share. |
| `basis` | `VARCHAR` | `grant_date_fair_value` or `appraisal`. |

### `equity_holder`

One row per award and holder interval. A new interval starts whenever the holder moves cost center or company, until the award settles.

| column | type | description |
|---|---|---|
| `award_id` | `BIGINT` | The award. |
| `employee_id` | `BIGINT` | The holder. |
| `cost_center` | `VARCHAR` | The holder's leaf cost center during the interval. |
| `company` | `VARCHAR` | The holder's company during the interval. |
| `valid_from` | `DATE` | When the interval began. |
| `valid_to` | `DATE` | When it ended: the next interval's start, or when the award settled at its last vest, its holder's exit, or the world's end. |

### `equity_forfeiture_estimate`

One row per award cohort and revision of its forfeiture estimate.

| column | type | description |
|---|---|---|
| `cohort` | `BIGINT` | The cohort: the year awards were granted. |
| `effective_date` | `DATE` | When the estimate takes effect: the cohort's first January 1, then every January 1 after. |
| `estimate` | `DOUBLE` | The share of the cohort's shares expected to forfeit. |

### `equity_expense`

A reference ASC 718 ledger, one row per award and month while it accrues, inside the world's window. It's computed by a plain loop per award, independently of the engine, so the engine's calculator can be scored against it.

| column | type | description |
|---|---|---|
| `award_id` | `BIGINT` | The award. |
| `period` | `DATE` | The month-end. |
| `gross` | `DOUBLE` | Straight-line expense per tranche, from the award month through the vest month, before forfeitures. |
| `net` | `DOUBLE` | Gross times one less the cohort's forfeiture estimate in effect. |
| `trueup_attrition` | `DOUBLE` | The reversal of everything recognized for a tranche, in the month it's forfeited. |
| `trueup_rate` | `DOUBLE` | When the estimate is revised, the gross accrued so far times the estimate's fall, for tranches still accruing. |
| `trueup_vesting` | `DOUBLE` | At a vest, the tranche's value less everything recognized for it. |
| `recognized` | `DOUBLE` | Expense recognized in the month: net plus the three true-ups. |

An award granted and settled inside the window recognizes exactly its vested shares times its grant value.

## Transaction codes

Every code has a disposition:

- **Measured** codes count toward the measures readers compute.
- **Excluded** codes are real flows, but they belong to the contingent pool, which headcount measures leave out.
- **Ignored** codes have no effect on headcount, and readers skip them.

A code missing from `dim_trans_type` is an error: readers refuse the data rather than guess.

| code | disposition | amount | what it records |
|---|---|---|---|
| `ACTIVE` | measured | 1 | A person in post at a month-end. |
| `HIRE` | measured | 1 | A regular employee's first day. |
| `TERM` | measured | -1 | A regular employee's departure. |
| `RIF` | measured | -1 | A departure in a termination program. |
| `TERM_PENDING` | measured | 0 | Notice of a departure, dated on the day of leaving and reported on the day it's given. |
| `XFER_OUT` | measured | -1 | Leaving a role for a transfer. |
| `XFER_IN` | measured | 1 | Arriving in a role by transfer. |
| `REORG_OUT` | measured | -1 | Leaving a cost center in a reorganization. |
| `REORG_IN` | measured | 1 | Arriving in a cost center in a reorganization. |
| `PROMO` | measured | 0 | A promotion, in place. |
| `RECLASS` | measured | 0 | A change of job family, in place. |
| `CONV_OUT` | measured | -1 | Leaving regular headcount for the contingent pool. |
| `CONV_IN` | measured | 1 | Joining regular headcount from the contingent pool. |
| `HIRE_REV` | measured | -1 | Reverses a hire recorded in error. |
| `TERM_REV` | measured | 1 | Reverses a departure recorded in error. |
| `HIRE_PENDING` | measured | 0 | An accepted offer, dated on the future start. |
| `CTG_START` | excluded | 1 | A contingent worker's start. |
| `CTG_END` | excluded | -1 | A contingent worker's end. |
| `ADMIN` | ignored | 0 | An administrative change with no effect on headcount. |
| `SYS_ADJ` | ignored | 0 | A system adjustment, found only in the aggregate fact. |

A move books two rows on one day: a leaving leg and an arriving leg. Transfers use `XFER_OUT` and `XFER_IN`, and reorganizations `REORG_OUT` and `REORG_IN`. A conversion out pairs `CONV_OUT` with `CTG_START`, and a conversion in pairs `CTG_END` with `CONV_IN`.

## The truth schema

The `truth` schema holds what only a simulator knows. Tests and recovery reports compare it with what models learn from the source tables.

`truth.manifest` has one row with a JSON document. Its keys:

- `generator_version`
- `contract_version`
- `config`: every parameter of the world
- `processes`: every random effect, regime, shock, batch, and program the run drew
- `quirks`: what each quirk did, such as the dropped snapshot dates

The tables hold the same truth in long form.

| table | columns | holds |
|---|---|---|
| `latent_plan` | `cost_center`, `month_end`, `target_headcount`, `annual_growth` | The latent growth plan that growth requisitions and plan vintages derive from. It runs past the world's end. |
| `plan_regime` | `vintage`, `cost_center`, `regime` | Each real cost center's regime in each vintage: `planned`, `flat`, or `decaying`. |
| `attrition_hazard` | `job_level`, `tenure_month`, `rating`, `monthly_hazard` | The monthly departure hazard by level, tenure, and rating (`unrated` included), before the season, the cost center's effect, and shocks. |
| `attrition_season` | `month_of_year`, `factor` | The seasonal factor per month of the year. It averages 1. |
| `attrition_shock` | `month_end`, `cost_center`, `factor` | The regime and org shock multiplier, for every month and cost center where it isn't 1. |
| `cost_center_effect` | `cost_center`, `process`, `scale`, `effect` | Each cost center's random effect: on attrition as a `multiplier`, and on backfill as a shift on the `logit` scale. |
| `requisition_hazard` | `channel`, `age_month`, `monthly_hazard` | Each channel's monthly hazard by a requisition's age in months: `external`, `transfer`, and `conversion` fills, and `unfilled` exits. The last age listed holds for every older one. |

A person's departure hazard in a month is the product of four parts, capped at 1:

- their entry in `attrition_hazard`;
- the month's season factor;
- their cost center's attrition effect;
- that month's shock factor, 1 if none is listed.

## Quirks

Real warehouses carry the marks of the systems that feed them. The world can add those marks, one toggle each, so readers are tested against them. Every quirk is off by default, and each has a test. A quirk changes only how the data is recorded, never what happened: the processes, the equity world, and every identity above hold with any quirk on.

| toggle | effect |
|---|---|
| `quirks.backdated_share` | This share of transactions reports 30 to 90 days after its event, rather than within the usual lag. Month-end stocks are never backdated. |
| `quirks.correction_share` | This share of hires and departures is first recorded on a wrong day. Each gets a reversal (`HIRE_REV` or `TERM_REV`) on that day, and is then recorded on the right day. Corrections net out within their month. |
| `quirks.no_show_pair_share` | For this share of offers that never start, a hire is recorded on the start date and reversed with `HIRE_REV` within the month. Requisitions see no difference. |
| `quirks.pending_hire_rows` | Each accepted offer also records a `HIRE_PENDING` row, dated on the future start and reported on the offer date. |
| `quirks.aggregate_only_share` | This share of cost-center months gains a `SYS_ADJ` row in the aggregate, with a row count but no amount, that no transaction stands behind. The code is ignored, so no measure moves. |
| `quirks.snapshot_gap_share` | This share of month-ends is missing from `req_snapshot`, as when a feed run is missed. The dropped dates are recorded in the manifest. |
| `quirks.state_seam_month` | From this month on (`YYYY-MM`), `req_snapshot` spells its state `OPEN` instead of `open`, as when a feed is rebuilt. |
| `quirks.cost_center_aliases` | The leaf cost center goes by another name in some tables. It's `department` in `req_interval`, `cost_centre` in `pending_start`, `org_unit` in `plan_headcount`, and `cc` in `equity_award` and `equity_holder`. `contract_columns` records the name each table uses. |
| `openings.bulk_share` | This share of growth requisitions are bulk placeholders with many openings, filled over several months. It's on by default. |
| `compensation.missing_range_share` | This share of `comp_range` cells is dropped. Award sizing treats a missing range as an error, and this lets tests confirm it does. It's off by default. |

Two properties of the data aren't toggles. They follow from how it's observed, and readers must handle both:

- **Requisitions opened and filled within a month** never appear in `req_snapshot`. Only `req_interval` holds them.
- **Closures are right-censored.** A requisition still open at the world's end has no closing date, and an award's tranches can vest after the end.

## Exports

`hcsim synth export` writes a world as Parquet. There's one file per source table and view, the truth tables under `truth/`, and the manifest as `truth/manifest.json`. Views become tables holding the rows they show. `hcsim synth import` builds a DuckDB warehouse back from an export. The `small` world, seed 0, is attached to each published release as such an export.

## Changing the contract

The version has a major and a minor part:

- An addition that existing readers can ignore raises the minor version. That covers a new table, a new column, or a new code with a disposition.
- Anything that breaks an existing reader raises the major version. That covers renaming or removing a table or column, changing a type, or changing what a value means.

A change to the contract updates this page, the version the world records, and the tests that keep the two in step, in the same change.
