---
name: design-cycle
description: Design a part of headcount-simulation, chiefly the explorer app, as a sequence of decisions, one per turn. For each decision, sketch the alternatives as an interactive widget, recommend one, wait for the user's pick, and record it in a design doc. Use when the user asks to design, explore or decide a screen, a demo, a chart, or a style question, or invokes /design-cycle.
argument-hint: <what to design, e.g. "the explorer's valves demo">
---

# Design cycle

Design **$ARGUMENTS** as a numbered sequence of decisions, each sketched as an interactive widget
with three or four alternatives, settled by the user, and written up as it's settled. This
follows the design cycles of the FretCodex project, adapted to a local web app.

## 1. Start: read, then lay out the sequence

1. **Read what's already decided** before proposing anything: `docs/design/overview.md`,
   `docs/design/notation.md`, `docs/glossary.md`, the ADRs in `docs/adr/`, the domain pages in
   `docs/domains/`, any `docs/design/*-Design.md` from earlier cycles, and the code the topic
   touches (for the explorer: `apps/explorer/src` and `src/hcsim/app`). Never reopen a settled
   decision; build on it, and say which earlier decision a new one leans on (for example "E2's
   left rail").
2. **Lay out the decisions in chat** as a table: a short ID (a letter for the topic and a number,
   such as E1 to E8), a one- or two-word name, and the question each answers. Order them so each
   builds on the last: purpose and shape before layout, layout before controls and charts, each
   demo after the shared parts, and edge states last.
3. **Track the effort** with a GitHub issue labelled `area:app`, or the one that exists. Scan its
   text first (section 4).
4. **Sketch the first decision in the same turn.** Don't write the doc until the first pick.

## 2. Each decision: one per turn

**Sketch it with the visual tool** (`show_widget`; load its guidance with `read_me` first, modules
`mockup` and `interactive`):

- **Three or four real alternatives,** each one a design someone could defend. Mark exactly one
  "(recommended)".
- **Draw the real thing:** the explorer in a desktop browser window at true proportions (a
  1280 by 800 viewport, scaled to fit), with the app's own colours and type (its stylesheet,
  `apps/explorer/src/styles.css`, until a cycle changes them). Put the drawing on a
  `var(--surface-1)` stage.
- **Toggles that test the options,** not decoration: light and dark; a wide window and a narrow
  one; the states that matter (at rest, running, an error, an empty answer); the data that
  stresses it (one cost center or a hundred, a month or thirty-six). For a numeric decision, make
  it an explorer with sliders and presets, and show the consequence live.
- **Real numbers.** Draw with the engine's own answers: run the demo's API on the tiny world
  (`hcsim.app.demos`, or `hcsim app serve` and `/api/...`) and embed what it returns, rather than
  inventing data. Numbers decide chart scales, label density, and what a run takes to answer.
- **One button per option** that calls `sendPrompt` with an exact message: "E3: go with B, a Run
  button." For a decision in parts, use a tab per part and one "Send these picks" button that
  sends every part's pick.

**In the chat text,** one short bullet per option: what it is, and what it costs. Then the
recommendation and why the others lose. Don't repeat what the widget already shows. Flag anything
the pick will force elsewhere (an API change, a slower demo, a new dependency).

**Widget pitfalls:**
- Wrap the script in an IIFE: a top-level `const` can throw and blank the widget.
- For dragging, keep the drag state outside the render function and listen for `pointermove` and
  `pointerup` on `window`: re-rendering replaces the element and drops pointer capture.
- Lay out labels with computed rects, never eyeballed offsets.
- Widget chrome text is 11 pt or larger; text inside the drawn browser can be smaller.

## 3. On a pick: record it at once

- **Take the pick exactly as given,** including combinations ("C's layout with D's colours").
- **If the user's wording can be read more than one way, ask** with `AskUserQuestion`, giving each
  reading as an option with a preview, rather than guessing. When they answer with a model of
  their own, restate it in a line before recording it.
- **Write it into `docs/design/<Topic>-Design.md`** (create it on the first pick):
  - a status line at the top, and a sequence table with each decision's status;
  - a section per decision: *Decided <date>: <summary>.*, then the details as bold-led bullets
    and tables, its accessibility, and its consequences;
  - "The code's part": what the decision needs built, in the frontend and the API;
  - **What was considered:** the question, and each rejected alternative with the reason it lost;
  - a Decisions table at the end, one row per decision.
- **Mark details the user didn't decide as *Proposed*,** and say so in the reply. Don't present an
  addition of your own as decided.
- **If the pick changes something already built,** fix the code in the same turn, with tests, and
  run `make check` and `make explorer-check`, and build the frontend.
- Then sketch the next decision.

## 4. Conventions

- On-screen labels use sentence case, as the explorer writes them ("The three valves").
- Prose in plain, short sentences: lists and tables over long paragraphs, and no parenthetical
  dashes.
- The repository is public and clean-room: generic vocabulary only, and synthetic values only.
  Scan any text meant for GitHub first: `uv run --quiet python tools/sanitize_scan.py FILE`.
- Design docs are prose, so their PR waits for the user's review. Commit only when the user asks,
  and keep code fixes and design docs in separate commits, with messages that list the decisions.
- When every decision is settled, mark the doc's status "decided, not built", and open an issue
  for each piece of the build, naming each decision's code part. Offer the build as a follow-up
  session with a brief that names those issues.
