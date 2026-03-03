# PCFG module summary (`pcfg.py`)

This document summarises the design and behaviour of `src/motifgen/pcfg.py`.

## Goal

The PCFG component generates a **high-level structural plan** (Vordersatz → Fortspinnung → Epilog) as a **sequence of timed structure events**.  
These events do **not** directly output notes; they are later **realised** into melody content by the motif realiser + k-gram Markov infiller.

In other words:

1. **PCFG** chooses *what kind of motivic actions happen, and roughly when*  
2. **Realiser/Markov** chooses *the actual notes* that implement those actions

## Key abstractions

### Symbols
- `Symbol = str`
- Nonterminals (expandable): e.g. `S, V, F, E, SPIN`
- Terminals (output tokens): e.g. `M0, REP, SEQ+1, SEQ+2, INV, RET, CAD`

A symbol is considered **terminal** if it has **no rules** in `rules_by_lhs`.

### Grammar rules

`GrammarRule(lhs, rhs, prob)`:

- `lhs`: a nonterminal to expand
- `rhs`: a tuple of symbols (terminals and/or nonterminals)
- `prob`: probability of choosing this rule (rules with the same `lhs` must sum to ~1.0)

### Events (scheduled output)

`Event(tok, start_units, dur_units)`:

- `tok`: terminal token (e.g. `M0`, `SEQ+2`, `CAD`)
- `start_units`: start time on a discrete grid
- `dur_units`: duration on that grid

**Time grid convention**
- `beats_per_bar = 4` (4/4)
- `units_per_beat = 2` → **unit = half-beat** (eighth-note grid)
- `units_per_bar = beats_per_bar * units_per_beat = 8`

Example: `dur_units=4` means **2 beats**.

### Section layout

`SectionSpec(v_bars, f_bars, e_bars)` defines how many bars are allocated to:
- `V` (Vordersatz)
- `F` (Fortspinnung)
- `E` (Epilog)

These must sum to `num_bars`.

### Sampling configuration

`SamplerConfig` contains guards and grid settings:
- `max_recursion_depth`: prevents runaway recursion (e.g. `SPIN → SPIN SPIN` repeatedly)
- `max_events_per_section`: prevents excessively long plans
- `beats_per_bar`, `units_per_beat`: grid definition
- optional `seed`

## How sampling works

### 1) Grammar expansion → terminal sequence

`PCFG.sample_terminals(cfg, rng)`:

- Starts with the start symbol, typically `S`
- Repeatedly finds the first nonterminal in the current sentential form
- Samples a rule for that nonterminal using weighted probabilities
- Replaces the nonterminal with the rule’s RHS
- Stops when only terminals remain

**Recursion guard**
- If `max_recursion_depth` is reached, any remaining nonterminals are replaced with a safe default terminal (currently `"M0"`).

Output: a flat list like:

```
["M0", "REP", "SEQ+2", "INV", "M0", "CAD"]
```

### 2) Terminal sequence → scheduled events

`PCFG.schedule_terminals(terminals, section, cfg, ...)`:

- Converts terminals into `Event`s with start times and durations
- Allocates a time window for each section (V/F/E) using `SectionSpec`
- Assigns each token a duration:
  - motif-like tokens: sampled from `motif_dur_units` using `motif_dur_probs`
  - `CAD`: typically forced to a longer duration (currently `max(motif_dur_units)`)
- Places events sequentially in each section until the section’s time budget is exhausted

**Important note about CAD**
If terminals are split naively across sections, `CAD` can end up in the wrong section and then be re-added at the end, causing duplicates.
Fix: ensure **exactly one** `CAD`, placed at the end of section `E` (by filtering out CAD everywhere and appending one final CAD).

### 3) One-shot plan sampler

`PCFG.sample_plan(num_bars, section=None, cfg=None, ...)`:

- Samples terminals from the grammar
- Chooses a default `SectionSpec` if none is provided (e.g. 2/5/1 for 8 bars)
- Schedules terminals into timed `Event`s

This is the main API used by `main.py` / the generation pipeline.

## Minimal MVP grammar

`make_minimal_mvp_grammar()` constructs a small PCFG:

- `S → V F E`
- `V → M0 REP | M0`
- `F → SPIN`
- `SPIN → SPIN SPIN | SEQ+1 | SEQ+2 | INV | M0`
- `E → CAD`

This produces:
- an initial presentation (V)
- a recursive “spinning-out” middle (F)
- a cadence/closure (E)

## Expected output format

The PCFG ultimately produces an ordered list of events, e.g.:

```python
[
  Event(tok="M0",    start_units=0,  dur_units=4),
  Event(tok="REP",   start_units=4,  dur_units=4),
  Event(tok="SEQ+2", start_units=8,  dur_units=4),
  ...
  Event(tok="CAD",   start_units=56, dur_units=4),
]
```

These events are later realised into melody notes and highlighted in the score output.

## Design intent

- **Interpretability:** The plan is human-readable (`M0`, `SEQ+2`, `CAD`, …).
- **Hierarchy:** Structure is created by grammar expansion, not only local statistics.
- **Separation of concerns:** PCFG handles *form*; Markov handles *local style / filling*.
- **Guards:** Depth and event caps keep sampling stable and prevent unbounded recursion.

---

## How to use the Rohrmeier & Neuwirth paper
	•	We model form as hierarchical grouping + formal functions + repetition structure using a small PCFG.  ￼
	•	We operationalise repetition structure via motif transform tokens (REP/SEQ/INV/etc.).
	•	We include anchor points via a cadence token CAD near section ends.

That is enough for your write-up to say: “Inspired by generative-grammar approaches to musical form (Rohrmeier & Neuwirth, 2025)…”.