# The explorer's design

**Status:** decided, not built (#647). All ten decisions are settled, and the build is tracked in #650.

The explorer is the local, interactive demo of the engine (M9.2), built on FastAPI and React
(decision D6). This doc records its design, one decision at a time, each sketched with real
alternatives and settled by the maintainer. The explorer as first built (#638) is the starting
point: four demos behind tabs, with sliders and SVG charts.

## Sequence

| ID | Decision | Question | Status |
|---|---|---|---|
| E1 | Purpose | Who is the explorer for, and what shape does that give it? | Decided |
| E2 | Layout | How are the chapters arranged and reached? | Decided |
| E3 | Controls | Where does a chapter's input live, and when does it run? | Decided |
| E4 | Chart language | One visual vocabulary for every chart: marks, bands, colour roles, light and dark | Decided |
| E5 | Sampling | How quota versus independent draws is shown | Decided |
| E6 | Valves | How a plan and its valves are shown | Decided |
| E7 | Lifecycle | How the requisition stock and its flows are drawn | Decided |
| E8 | SBC | How grants become expense | Decided |
| E9 | Provenance | What the page says about the run behind it | Decided |
| E10 | Edge states | Slow runs, errors, narrow screens, the keyboard, and screen readers | Decided |

## E1. Purpose: a guided story

*Decided 2026-10-25: the explorer is a guided story. It's one scrolling page in four chapters,
each explaining one idea with a live chart from the engine.*

- **One page, four chapters, in order:**
  1. quota sampling;
  2. the three valves;
  3. the requisition lifecycle;
  4. the SBC layer.
- **Each chapter teaches one idea.** A few lines of explanation, then a live chart drawn from a
  real run on the world, then one or two controls to change it, then a line that hands on to
  the next chapter.
- **The story order is the engine's order of ideas.** Sampling explains how a guided month
  realizes its counts. The valves build on it to steer toward a plan. The lifecycle shows the
  requisitions those valves open and fill. SBC turns the resulting grants into expense.
- **Audience** (*Proposed*): someone meeting the engine for the first time, including readers
  arriving from the project's blog posts and docs. Modelers who want to compare runs have the
  studies and their scorecards.

**Accessibility** (*Proposed*, for E10): each chapter is a section with its own heading, so the
page reads in story order with a screen reader. Each chart carries a one-line text summary of
what it shows.

**Consequences:**
- E2 decides how the chapters are reached and how the page scrolls, rather than tabs.
- Each chapter needs a few lines of prose, which go to the maintainer's review like every doc.
- The tab bar goes, and each demo becomes a chapter's body.

**The code's part:** `App.tsx` renders the four chapters in order in place of the tabs. Each
chapter wraps one existing demo with its introduction and its hand-on line. The API doesn't
change: each chapter reads one existing `/api` demo.

**What was considered:** what is the explorer for, and so what shape should it take?

- **B. Demo tabs**, as first built. Any demo is one click away, but nothing explains why a demo
  matters or ties one to the next.
- **C. A workbench**: a panel for the world and parameters, and a Run button that compares two
  runs. It suits modelers, but it needs API work to compare runs, offers a newcomer no way in,
  and repeats what studies and scorecards already do.
- **D. A dashboard**: all four demos as cards on one screen. Everything is in view, but each
  chart is too small to show the differences that matter, and there's no room to explain them.

## E2. Layout: a chapter rail

*Decided 2026-10-25: a sticky chapter rail. It lists the four chapters, marks the one in view
as the reader scrolls, and jumps to a chapter on a click. In a narrow window it folds into a top
bar that names the chapter and opens the list.* It builds on E1's single scrolling page.

- **Wide windows:** the rail sits on the left and stays in place as the page scrolls. It holds
  the explorer's name and the four chapters, numbered, with the chapter in view marked.
- **Narrow windows:** the rail folds into a bar across the top that reads "2 of 4: The three
  valves" and opens the chapter list.
- **Clicking a chapter** scrolls to it, and the page's address names it, so a link can open the
  explorer at a chapter.
- **Anchors** (*Proposed*): `#sampling`, `#valves`, `#lifecycle`, and `#sbc`.
- **The text column** (*Proposed*): about 620 px, roughly 75 characters a line at the body size.
  Charts take the column's width.
- **Where the rail folds** (*Proposed*): below a 960 px window.

**Accessibility** (*Proposed*, for E10): the rail is the page's navigation landmark, and the
chapter in view carries `aria-current`. The narrow bar is a button that opens the list, and the
keyboard reaches every chapter in order.

**Consequences:**
- E4's charts must read well at the text column's width.
- Every chapter has a stable anchor, which E9 can reuse to cite a chapter's run.

**The code's part:** a `ChapterRail` component, and an observer that marks the chapter in view as
the reader scrolls. Each chapter is a `section` with its anchor, and a narrow-window bar replaces
the rail below the fold width.

**What was considered:** how a reader finds their place and moves between chapters.

- **B. A progress strip** of four steps across the top. In a narrow window its titles shrink to
  numbers, and it stops saying where you are.
- **C. A stepper** that shows one chapter per screen, with Previous and Next. A reader can't
  scroll ahead or glance back, and it breaks E1's one-page story.
- **D. Contents only**, a list at the top and no navigation once you scroll. A reader deep in
  the page has no sign of where they are or how much is left.

## E3. Controls: presets, then adjust

*Decided 2026-10-25: each chapter offers three named scenarios that make its point, fetched when
the page loads, so a click answers at once. "Adjust" opens live sliders for anyone who wants to
go further.* It builds on E1's chapters and sits under each chapter's chart, in E2's text column.

- **Three presets per chapter**, as buttons under the chart, one of them selected. Each names a
  scenario rather than a number, so the reader sees what it means before they pick it.
- **Fetched at load:** every chapter's presets are asked for as the page loads, and the engine
  keeps each answer, so a preset answers at once.
- **"Adjust"** opens the chapter's sliders. They run the engine once a drag settles, and the
  chart shows a running state until the answer arrives.
- **Each chapter's presets** (*Proposed*):

  | Chapter | Presets |
  |---|---|
  | Quota sampling | Even rows; uneven rows; very uneven rows (spread 0, 1, 2) |
  | The three valves | No growth; a modest plan, 5%; an aggressive plan, 20% |
  | The requisition lifecycle | Six months; a year; three years |
  | The SBC layer | Six months; a year; three years |

- **Selected at load** (*Proposed*): the middle preset of each chapter.
- **The sliders' debounce** (*Proposed*): 250 ms after the last change.

**Accessibility** (*Proposed*, for E10): the presets are a radio group with the chapter's name as
its label. "Adjust" is a disclosure button that says whether it's open, and the running state is
announced politely.

**What it costs:** about 2 s of engine time at load on the tiny world, mostly the valves
chapter's three plans. A larger world takes longer, which E10 covers.

**Consequences:**
- Each chapter's preset labels are part of its prose, and go to the maintainer's review.
- The API doesn't change: a preset is a named set of the parameters each demo already takes.

**The code's part:** a `Presets` component (a radio group of named parameter sets), an
`Adjust` disclosure holding the existing sliders, and a loader that fetches every chapter's
presets when the page loads.

**What was considered:** where a chapter's input lives, and when it runs.

- **A. Live sliders** alone. Fast chapters feel instant, but the valves chapter shows a running
  state for about half a second on every new value, and a newcomer gets bare sliders with no
  sense of what's worth trying.
- **B. A Run button.** Nothing runs by accident, but the chart can disagree with the sliders above
  it until the reader presses Run, which confuses a story.
- **D. Controls in the rail.** The chart gets the column to itself, but its controls sit apart
  from it, and in a narrow window they hide behind the top bar's menu.

## E4. Chart language: quiet, with one accent

*Decided 2026-10-25: every chart is quiet, with one accent. What the chapter is about takes the
accent colour, with its spread across trajectories as a band behind it. Everything else is
context, in gray, and each line is labelled at its end rather than in a legend.* It builds on E1's
one point per chapter and on E2's text column.

- **One subject per chart:** the run, the series or the bar set that makes the chapter's point.
  It's drawn in the accent, at about 2.4 px.
- **Context in gray:** the other runs, as thin gray lines at about 1.2 px, and the plan as a
  dashed line in the text colour.
- **A band for the spread:** the subject's 10th to 90th percentile across trajectories, as a
  light tint of the accent, with its median as the line.
- **Labels at line ends**, in place of a legend: the subject's label is bold and in the accent,
  and the others are in the muted text colour. Labels that would collide are nudged apart.
- **Bars follow the same rule:** the subject's bars take the accent, and a comparison takes gray.
- **Colour roles**, as tokens for light and dark (*Proposed*):

  | Token | Light | Dark | Used for |
  |---|---|---|---|
  | `--ink` | `#111827` | `#e5e7eb` | text, the plan |
  | `--muted` | `#6b7280` | `#9ca3af` | axis labels, context labels |
  | `--grid` | `#e5e7eb` | `#374151` | gridlines |
  | `--context` | `#878f9b` | `#6b7280` | context lines and bars |
  | `--accent` | `#2563eb` | `#60a5fa` | the subject, its band (at 18%) |
  | `--surface` | `#ffffff` | `#111827` | the chart's background |

- **The theme follows the system** (*Proposed*): the page reads `prefers-color-scheme`, with no
  toggle of its own.
- **The spread's statistic** (*Proposed*): the median as the line, rather than the mean, so the
  line always sits inside its band.
- **Trajectories per preset** (*Proposed*): 10, enough for a stable 10th and 90th percentile on
  the tiny world.

**Accessibility** (*Proposed*, for E10): colour is never the only cue. The subject is also the
thickest line and its label is bold, and the plan is dashed. Every line colour meets 3:1 against
the chart's background: the accent at 5.2:1 in light and 7.0:1 in dark, and the context gray at
3.3:1 and 3.7:1. Each chart's text summary names its subject and the band's range.

**What it costs:** 10 trajectories per preset take about 4 s of engine time at load on the tiny
world, up from about 2 s at 2 (E3's figure). The valves chapter goes from 1.3 s to 1.7 s, and the
SBC chapter from 0.4 s to 1.8 s.

**Consequences:**
- **The API changes.** The valves, lifecycle and SBC demos return a 10th, 50th and 90th percentile
  per month across trajectories, where they return a mean today. This revises E1's and E3's "no
  API change".
- E5 to E8 each name their chapter's subject and its context.
- The dark theme needs its own accent, since today's `#2563eb` is too dark against a dark
  background.

**The code's part:** `charts.tsx` gains a band mark and end labels, and loses the legend and
`COLOURS`. The colours become CSS custom properties in `styles.css`, set for light and dark. Each
demo in `demos.py` returns its percentiles, and the frontend asks for 10 trajectories.

**What was considered:** one visual vocabulary for every chart. The sketch drew the valves
chapter's real headcount over 20 trajectories at 5% growth on the tiny world.

- **B. A colour per series**, with a legend, as first built. Four saturated lines compete
  equally, demand ending near 285 and rate calibration near 334, and nothing shows how much a run
  varies.
- **C. Small multiples**, one panel per run on shared axes. Each run is legible, but four panels
  at the column's width are about 150 px each, and demand's first-month drop shrinks to a few
  pixels.
- **D. Fans everywhere**, every run's band and median at once. The bands of no alignment and
  supply overlap almost entirely from month 1 to 3, between about 312 and 326, and become a wash.

## E5. Sampling: the count, then who

*Decided 2026-10-25: the sampling chapter shows two charts. The first shows how many rows each
trajectory drew. The second shows how often each row was drawn, against the model's probability.
Together they say quota sampling fixes the count and lets the model decide who.* It applies E4's
language: quota sampling is the subject, and independent draws are its context.

- **How many rows each trajectory drew:** a histogram over 500 trajectories. Independent draws
  are gray bars spread around the expected count. Quota sampling is one accent bar at the quota,
  drawn narrower and labelled with how many trajectories it holds.
- **How often each row was drawn:** each row's share of the trajectories that drew it, likeliest
  row first. Quota sampling is the accent line, independent draws the gray line, and the model's
  probability the dashed reference line, in the text colour as E4's plan is.
- **The chapter's text says what the second chart shows:** quota sampling draws the rows a model
  rates higher more often, but not in exact proportion. On very uneven rows, a row the model
  rates as certain is drawn 71% of the time, where independent draws always draw it. That's a
  property of drawing rows one at a time without replacement, measured exactly in
  `model/ml/inclusion.py`.
- **The histogram's height** (*Proposed*): capped at 80 trajectories so the independent spread
  reads clearly. The quota bar runs off the top with a break mark and its count, "all 500".
- **The rows' grouping** (*Proposed*): 20 groups of 10 rows, each group's mean share, so the lines
  are smooth. The frontend groups them from the 200 rates the API already returns.
- **The rate lines' labels** (*Proposed*): at their left ends, where the lines differ most, since
  all three meet near zero at the right.

**Accessibility** (*Proposed*, for E10): the quota bar is narrower than the gray ones and
labelled, so it doesn't rely on colour. Each chart has a one-line summary, such as "Quota sampling
drew 20 rows in all 500 trajectories; independent draws took 8 to 31."

**What it costs:** nothing new. The demo already returns the histograms and the rates, and runs a
preset in about 14 ms.

**Consequences:**
- This chapter has two charts where the others have one, so it's the longest, and E10's narrow
  window stacks them.
- The sentence on the gap between the two samplers is new prose and goes to the maintainer's
  review.
- The API doesn't change.

**The code's part:** `SamplingDemo` draws the histogram with the clipped quota bar and the grouped
rate chart in E4's language. The grouping happens in the frontend.

**What was considered:** how quota versus independent draws is shown. The sketch drew 200 rows and
a quota of 20 at each of E3's presets, from the real samplers.

- **A. The count only:** the histogram alone. It shows how many but never who, and the three
  presets look almost the same.
- **C. Draws as strips:** ten trajectories per sampler, one strip of rows each, with its count at
  the end. It shows both halves in one chart, but needs the API to return example draws, and shows
  the gap between the samplers only faintly.
- **D. Draw by draw:** a "Draw again" button that adds one trajectory at a time. It's hands-on,
  but the spread only shows after a dozen clicks.

## E6. Valves: one chart, a valve at a time

*Decided 2026-10-25: the valves chapter draws one chart, with a chip per valve. The chosen valve is
the subject, with its band, and the plan and the other runs are its context. A line under the
chart names the valve's lever and where it ends against the plan.* It applies E4's language to
E3's three plans.

- **The chart:** headcount at each month-end, every cost center together, for six months. The
  plan is the dashed line, the chosen valve is the accent line with its band, and no valve and
  the other valves are thin gray lines. Every line is labelled at its end.
- **The chips** choose the valve in focus: No valve, Supply, Flow, Rate. They change only the
  focus, so they answer at once and don't run the engine. The presets choose the plan.
- **The line under the chart** names the lever and the ending. For example: "Rate calibrates the
  domains' rates, so an unguided run drifts toward the plan. It ends at 331.5, 4.5 under the
  plan's 336."
- **The valves' names** (*Proposed*): the plan's names for them (supply, flow, rate), with the
  method behind each in the chapter's text, rather than the methods' own names.
- **The valve in focus at load** (*Proposed*): Supply, the first, so the chips read left to right
  in the order the chapter's text explains them.
- **The table of biases goes** (*Proposed*): the line under the chart gives the ending, and the
  bias per cost center stays in the studies' scorecards.

**Accessibility** (*Proposed*, for E10): the chips are a radio group labelled "Valve in focus".
The line under the chart is the chart's text summary, announced politely when the focus changes.

**What it costs:** nothing beyond E4. Its 10 trajectories for each method and plan take about
1.7 s of engine time at load on the tiny world.

**Consequences:**
- The chapter has two controls: E3's presets choose the plan, and the chips choose the focus.
- **The flow valve's first-month loss is fixed (#648).** The sketch showed it losing about 33
  heads in month 1 in every preset, even against a flat plan. Alignment was re-deriving each
  month's flows from reconciled ones, and the engine could realize the exits but not the hires.
  With the fix, it ends 4.5 under a 5% plan (331.5 against 336). Two smaller gaps stay open on
  #635: a 20% plan outruns the openings, and a flat plan slides about 4 heads over six months.
  The chapter's text says so if they're still open when it's built.
- Supply's slow start is real, not a fault. Its requisitions take months to fill, so at 5% it
  ends below no valve (315 against 319.5), and only rises at 20%. The line under the chart says so.

**The code's part:** `ValvesDemo` gains a radio group for the focus and the line under the chart,
and loses its table. The API returns E4's percentiles for each method.

**What was considered:** how a plan and its valves are shown. The sketch drew 10 trajectories of
each method on the tiny world, against plans rising 0%, 5% and 20% over six months.

- **B. Scroll through the valves:** the chart stays at the top of the chapter while a paragraph per
  valve scrolls under it, each putting its valve in focus. It reads well, but needs a sticky chart
  inside a chapter and a second scroll observer, and a narrow window has little room for both.
- **C. The gap to the plan:** the chart's zero line is the plan, and each line is its run's
  headcount minus the plan. Misses are easy to compare, but the plan's rise disappears, and a
  newcomer has to read "below zero" as "short of the plan".
- **D. Where each valve ends:** one row per valve, a dot and a whisker for its gap at month 6. It
  compares fastest, but loses the paths, which are where the valves differ most.

## E7. Lifecycle: flows through a stock

*Decided 2026-10-25: the lifecycle chapter draws a flow diagram over the preset's horizon. Growth
and backfill flow into the stock of open openings, and fills and unfilled exits flow out, each
ribbon as wide as its total. The stock's line and band sit under it.* It applies E4's language:
the stock is the subject, and the flows are its context.

- **The diagram:** the openings opened on the left, the open openings in the middle, and the
  openings closed on the right. Each ribbon is as wide as its total over the horizon, in gray,
  and labelled with its name and total.
- **The stock** is the accent box, labelled with its count at the start and at the end.
- **The identity, in words, above it:** for a year on the tiny world, "46.7 opened = 38.4 closed
  + 3.0 more open". The widths balance the same way.
- **The stock's line and band** under the diagram give the months: open openings at each
  month-end, the median of the trajectories with E4's band.
- **Channels with nothing to carry** are named in one quiet line under the diagram, rather than
  drawn as empty ribbons. On the tiny world that's six of the ten flows: campus openings,
  backfill for transfers, and the conversion, transfer, campus and pending-start fills.
- **The ribbons' scale** (*Proposed*): set per preset so the diagram keeps its size, with each
  ribbon's total printed beside it. The totals are what compare across presets, not the widths.
- **Openings opened and closed are split** (*Proposed*): growth and backfill on the left, filled
  and unfilled on the right. Each fill channel gets its own ribbon when a world fills through
  more than one.

**Accessibility** (*Proposed*, for E10): the diagram's text summary is its identity line, and each
ribbon's name and total are text, not only width. The flow diagram is one image with that summary,
and the stock's chart below has its own.

**What it costs:** nothing beyond E4. The frontend sums the monthly means the API already returns,
and the stock's band comes from E4's percentiles.

**Consequences:**
- The tiny world shows only four of the ten flows, so the diagram is simpler there than the
  engine is. Whether the explorer runs on a larger world by default is for E9 or E10.
- A world that does use every channel needs room for five ribbons on the right, so the diagram's
  height grows with the channels it draws.

**The code's part:** a `FlowDiagram` component (ribbons as SVG paths, sized from the totals) in
`charts.tsx`, and `LifecycleDemo` draws it over the stock's chart. Its flow bar chart goes.

**What was considered:** how the requisition stock and its flows are drawn. The sketch drew 10
trajectories of the tiny world over six months, a year and three years.

- **A. Stock over flows:** the stock's line, with a bar per month under it, opened above the line
  and closed below. The months read clearly, but the reader has to add the bars up to see why the
  stock moves, and at three years the bars are thin.
- **C. In and out, added up:** two running totals, opened on top of the starting stock and closed,
  with the stock as the shaded gap between them. Exact, but a newcomer has to learn to read a gap
  as a stock.
- **D. One month's ledger:** a month picked on the stock's line, with its identity written out in
  numbers. Concrete, but one month at a time, too narrow to carry the chapter.

## E8. SBC: net and true-ups

*Decided 2026-10-25: the SBC chapter draws a bar per month, split into the parts the books record:
net expense, the vesting true-up, and the attrition true-up below the line, with the recognized
total as the accent line.* It applies E4's language: the recognized total is the subject, and its
parts are context.

- **The bars:** net expense from the line up, in a light gray, then the vesting true-up stacked
  on it in a darker gray. The attrition true-up hangs below the zero line in the lightest gray.
  The rate true-up is stacked with the rest when a world has one.
- **The line:** the expense recognized each month, in the accent, with its band. Each part is
  labelled at the right.
- **The chapter's text names the parts before the chart:**
  - **net expense:** each tranche's value spread from its award to its vest, less the expected
    forfeitures;
  - **the vesting true-up:** at a vest, the gap between the value and what was recognized;
  - **the attrition true-up:** a leaver's unvested tranches reverse what they recognized;
  - **the rate true-up:** when the forfeiture rate is revised.
- **The quarterly spikes are the point, not noise.** The vesting true-up spikes in the months
  tranches vest in bulk: on the tiny world, months 0, 3, 6 and 9 of the year, up to 99k. The
  chart shows where each spike comes from.
- **Over the year on the tiny world:** recognized expense averages 256k a month. That's net 261k,
  plus vesting true-ups of 33k, less attrition true-ups of 38k.
- **The line and the parts** (*Proposed*): the bars are means across trajectories, so they add up
  to the line exactly. The line is therefore the mean, not E4's median, and its band is still the
  10th to the 90th percentile.
- **The rate true-up** (*Proposed*): drawn only when it isn't zero. On the tiny world the
  forfeiture rate is flat, so it's zero, and a line under the chart says so.
- **Gone from the chapter** (*Proposed*): the grants-by-award-type chart (one bar on the tiny
  world, all new hires) and the example grant's vesting schedule.

**Accessibility** (*Proposed*, for E10): the parts differ in position and shade as well as colour,
and each is labelled. The chart's text summary gives the year's averages, as above.

**What it costs:** a small API change. The SBC demo returns the monthly means of net expense and
of each true-up, and E4's percentiles of the recognized total, where it returns expense by seed
and sim today. It already runs about 1.8 s at load for 10 trajectories.

**Consequences:**
- The chart no longer splits expense by seed and sim, so the handover from inherited grants to the
  run's own goes into the chapter's text as a sentence.
- The chapter now explains four accounting terms before its chart, so its prose carries more of
  the load, and goes to the maintainer's review.
- A world with a revised forfeiture rate, or with grants of more types, draws more parts. Which
  world the explorer runs on is for E9 or E10.

**The code's part:** `SbcDemo` draws a diverging stacked bar chart with the recognized line, a new
`StackedBars` in `charts.tsx`. `demos.py`'s SBC demo returns the parts and the total's percentiles,
and drops the grants by type and the example grant.

**What was considered:** how grants become expense. The sketch drew 10 trajectories of the tiny
world over six months, a year and three years, and one real new-hire grant of 78 shares in 13
tranches.

- **A. Seed and sim:** one chart, expense from grants made before the run falling in gray, and
  from the run's own grants rising in the accent. It shows the handover but not where expense
  comes from.
- **B. One grant, then the whole:** the example grant's tranches and the cost they add up to,
  front-loaded so 60% lands in its first year, then A's chart. It explains a grant well, but not
  the true-ups that make each month's total jagged.
- **D. Follow one grant:** a month slider over one grant, filling each tranche as its cost
  accrues. Hands-on, but it never shows the run.

## E9. Provenance: a caption under every chart

*Decided 2026-10-25: every chart ends in one muted line that names the run behind it: the world,
its size, the month the run starts, the trajectories, the seed, and the engine's version.* A chart
carries its provenance wherever it goes, so a screenshot in a blog post still says what made it.

- **The caption:** for example, "Run on tiny5.duckdb (320 employees, 10 cost centers) from 2026-01
  · 10 trajectories · seed 7 · hcsim 0.1.0.dev0".
- **The sampling chapter's caption** says its rows are synthetic, not the world's: "200
  synthetic rows · 500 trajectories · seed 0 · hcsim 0.1.0.dev0".
- **Where the facts come from:** `/api/health`, read once at load, adds the engine's version and
  the world's size to the world's name and run date it returns today. Each chapter adds its own
  trajectories and seed.
- **The caption's place** (*Proposed*): inside the chart's frame, under the line E6 adds or the
  chart's own summary, separated by a hairline, in the muted text colour at the smallest size.
- **The world is named by its file** (*Proposed*): the world records no metadata saying which
  preset and seed made it, so the caption can't say "tiny, seed 5". Recording that is a change to
  the synthetic world, outside the explorer.

**Accessibility** (*Proposed*, for E10): the caption is text in the chart's figure, not part of
the image, so it's read after the chart's summary. The muted colour meets 4.5:1 (4.8:1 in light
and 7.0:1 in dark).

**What it costs:** a small API change: `/api/health` returns the version and the world's employee
and cost-center counts.

**Consequences:**
- E2's anchors and E9's captions together make any chart citable: a link to its chapter, and the
  run that drew it.
- The run's full settings, a profile anyone could rerun, aren't on the page. A "how this was made"
  disclosure (option C) can be added under the caption later without changing it.

**The code's part:** a `Provenance` caption component that every chart's frame ends with,
reading the facts from a page-wide context filled from `/api/health`. `server.py`'s health route
adds the version and the world's size.

**What was considered:** what the page says about the run behind it. The sketch showed the valves
chapter in the page, with the tiny world's real facts.

- **B. One run card in the rail:** the page's facts once, at the rail's foot. The charts stay
  clean, but a screenshot loses its provenance, and in a narrow window the card hides in the top
  bar's menu.
- **C. "How this was made" per chart:** a disclosure with the exact study profile and the `hcsim
  study run` command. Modelers could rerun any chart, but it needs a route that writes each
  preset's profile, and the valves profiles need their plan files served too.
- **D. A closing section:** "About these runs", a fifth section with a table of each chapter's
  settings. Everything is in one place, but a long scroll from the chart it explains.

## E10. Edge states: keep every answer on disk

*Decided 2026-10-25: every chapter shows its own waiting state while its presets run, and every
preset's answer is kept on disk. A later start, or a cache warmed ahead with `hcsim app prepare`,
loads at once.* It builds on E3's presets, a small fixed set of parameters that suits a cache.

- **The first start:** each chapter's chart frame says what the engine is doing and about how long
  is left, with a thin progress bar, and fills in as its presets arrive. Chapter 1 reads at once.
  The rail marks a chapter still running.
- **One lock per demo:** a slow chapter no longer holds up the others. The demos shared one lock
  when this was decided, so the lifecycle and SBC chapters would have queued behind the valves.
- **Kept on disk:** each preset's answer is the API's response, kept under a key of the world's
  contents (a hash of the file), the engine's version, the demo, and its parameters. A changed
  world or a new version misses the cache and runs again.
- **`hcsim app prepare --world W`** runs every chapter's presets, keeps them, and exits, so a
  first reader never waits. `hcsim app serve` reads the same cache.
- **A failed run stays in its chapter:** the chart frame shows the server's message and a "Try
  again" button, and the other chapters carry on.
- **No server:** if `/api/health` doesn't answer at load, the page says the server isn't running
  and gives the command that starts it.
- **The cache's place** (*Proposed*): `~/.cache/hcsim/explorer`, or `--cache DIR` on both commands.
- **Development versions** (*Proposed*): a version ending in `.dev` doesn't change with every
  commit, so the server reads the cache only when `--cache` is given. A code change then never
  serves a stale answer.
- **The time left** (*Proposed*): estimated from the bench's capacity model for the world's size
  (`hcsim.bench.capacity`), and rounded ("about 40 s").
- **Adjust stays live** (*Proposed*): its answers are kept in memory only, so the disk holds the
  presets alone. On a chapter whose presets took over 5 s each, the sliders run on release rather
  than after a 250 ms pause, and say how long a run takes.

**Edge states, gathered from E1 to E9** (*Proposed*):

| Concern | What the page does |
|---|---|
| Screen readers | Each chapter is a `section` with its heading (E1). The rail is the navigation landmark, with `aria-current` (E2). Every chart has a one-line text summary (E1, E5 to E8) and its caption as text (E9). |
| Live changes | The running state, and E6's line under the chart, are announced politely (E3, E6). A failed run is an alert. |
| The keyboard | Everything is reachable in reading order. Presets and E6's chips are radio groups that arrow keys move through, and "Adjust" is a disclosure (E3, E6). Focus shows as a ring in the accent. |
| Colour | Never the only cue: the subject is thickest and its label bold, the plan dashed, and parts labelled (E4, E5, E8). Lines meet 3:1 and text 4.5:1 in both themes (E4, E9). |
| Narrow windows | Below 960 px the rail folds into the top bar (E2). Charts take the column's width and stack, and keep their end-label gutter, so nothing scrolls sideways down to 320 px. |
| Theme and motion | The theme follows the system (E4). Nothing animates but the progress bar, which stays still under `prefers-reduced-motion`. |

**What it costs:** with one lock per demo, the first start takes as long as the slowest chapter.
Measured on the tiny world: about 2 s, the SBC chapter's three presets. On `small`, after #649
made fitting guidance take a quarter of a second, about 40 s for the SBC chapter (estimated from
its 12-month run, 8.0 s). Later starts read the kept answers in milliseconds.

**Consequences:**
- A new command, `hcsim app prepare`, and a `--cache` option. The explorer's docs page says to
  prepare before showing it.
- E3's and E4's load times (about 2 s and 4 s on the tiny world) are now first-start figures, and
  apply once per world and version.
- Hashing the world at startup costs a few seconds for a large file, once per start.

**The code's part:**
- `src/hcsim/app/cache.py`, the kept answers. `Explorer` takes one lock per demo and reads and
  writes through the cache.
- `cli.py` gains `prepare` and `--cache`. `/api/health` gains each chapter's estimated time.
- In the frontend, a `ChapterState` wrapper gives each chapter its waiting, failed, and ready
  states, and the app has a no-server state.

**What was considered:** how the page behaves while its runs are slow, and when they fail. The
sketch used real engine times on tiny and small. Before #649, small's valves chapter took about 28
minutes, almost all of it fitting guidance.

- **A. Each chapter as it's ready:** the same waiting states, with nothing kept between starts. Every
  start pays again: half a minute on small after #649, and 28 minutes before it.
- **B. Wait for the whole page:** one waiting screen, then the full story. The reader waits for the
  slowest chapter before seeing anything.
- **D. Stay on tiny:** refuse worlds over about 1,000 employees and keep everything live. Simplest,
  and small adds no new channels, but it caps the explorer for good.

## The build

Tracked in #650, one issue per piece, each naming the decisions it builds:

| Issue | Builds | Depends on |
|---|---|---|
| #651 | E4: the chart language, with percentiles from the API | |
| #652 | E1, E2: one page in four chapters, with the chapter rail | |
| #653 | E3: presets, then Adjust | #652 |
| #654 | E5: the sampling chapter | #651 |
| #655 | E6: the valves chapter | #651 |
| #656 | E7: the lifecycle chapter | #651 |
| #657 | E8: the SBC chapter | #651 |
| #658 | E9: a provenance caption under every chart | #651 |
| #659 | E10: kept answers, `hcsim app prepare`, and the edge states | #653 |
| #660 | The chapters' prose and preset labels, for the maintainer's review | |

## Decisions

| ID | Decision | Summary |
|---|---|---|
| E1 | Purpose | A guided story: one page in four chapters, each a short explanation with a live chart and a couple of controls |
| E2 | Layout | A sticky chapter rail that marks the chapter in view, folding into a top bar in a narrow window |
| E3 | Controls | Three named presets per chapter, fetched at load; "Adjust" opens live sliders |
| E4 | Chart language | Quiet, with one accent: the subject and its 10th to 90th percentile band in the accent, context in gray, labels at line ends |
| E5 | Sampling | The count, then who: a histogram of each trajectory's count, then each row's share of draws against the model's probability |
| E6 | Valves | One chart, a valve at a time: chips choose the valve in focus, and a line under the chart names its lever and its ending |
| E7 | Lifecycle | Flows through a stock: a flow diagram of openings opened, open and closed over the horizon, over the stock's line and band |
| E8 | SBC | Net and true-ups: a bar per month split into net expense and the true-ups, with the recognized total as the line |
| E9 | Provenance | A caption under every chart naming its run: the world and its size, the start, the trajectories, the seed, and the engine's version |
| E10 | Edge states | Keep every answer on disk: per-chapter waiting and failed states on a first start, kept answers after, and `hcsim app prepare` to warm them |
