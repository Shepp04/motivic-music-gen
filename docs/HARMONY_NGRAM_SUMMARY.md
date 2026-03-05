# Harmony n-gram module summary (`ngram.py`)

This document summarises the design and behaviour of `src/motifgen/ngram.py`, which builds a **two-stage harmony model** at **half-bar** resolution (2 harmony tokens per bar).

## Goal

Given harmony labels extracted from the Bach chorale corpus (Roman numeral labels), the module learns:

1. A **function-level n-gram** over coarse harmonic functions:
   - `T` (tonic), `PD` (predominant), `D` (dominant)
2. A **realiser distribution** `P(RN | Function)` to turn each function token into a specific Roman numeral chord label.

At generation time:
- sample a **function plan** of length `2 * num_bars`
- enforce a simple cadence (default: `D → T`) at the end
- sample a matching **Roman numeral plan** in parallel

The resulting chord plan is used later during melody realisation (e.g., prefer chord tones on strong beats).

---

## Data representation

### Input (from `dataset.py`)
Each processed piece is expected to include:

- `item["harmony_tokens"]`: Roman numeral labels at **quarter-note** (crotchet) resolution, length ≈ piece duration in quarters.
  - Example values: `["I", "I", "V", "V7", "N", ...]`
  - `"N"` means no label / unknown.

### Output plan (this module)
Half-bar plans (2 tokens per bar):
- Function plan: list of `["T","PD","D"]` of length `2 * num_bars`
- RN plan: list of simplified RN strings of length `2 * num_bars`

---

## Half-bar compression

`compress_rn_quarters_to_halfbars(rn_quarters)`:

- Assumes 4/4 (1 bar = 4 quarters)
- Groups quarter RN tokens into chunks of 2 (half-bar = 2 beats)
- Uses a **majority vote** per chunk to reduce noise from `chordify()` / non-chord tones

Output length: `floor(len(rn_quarters)/2)`.

---

## Roman numeral simplification (recommended)

A simplification step can be applied to keep RN vocabulary small and stable:

Allowed figures (MVP):
- Triads: root (`""`), first inversion (`63`)
- 7ths: root (`7`), third inversion (`42`), second inversion (`43`)

Everything else is canonicalised to one of these.  
This avoids noisy figures like `Ib753` or `VI6#42` produced by chordification/RN parsing.

(Implementation: `simplify_rn_figure(rn: str) -> str`.)

---

## Function mapping

`rn_to_function(rn)` maps a (simplified) RN string to one of:

- `T` (tonic / tonic prolongation)
- `PD` (predominant)
- `D` (dominant)
- `UNK` (unknown; dropped)

Robust MVP mapping:
- `V*`, `vii°*` → `D`
- `ii*`, `IV/iv*` → `PD`
- `I/i*`, `vi*`, optionally `iii*` → `T`
- `"N"` / parsing failure → `UNK`

---

## Generic n-gram model

### Training
`train_ngram(seqs, cfg)`:

- Builds a vocabulary from training sequences
- Collects counts for orders `1..k`
- Returns an `NGramModel` with:
  - `counts_by_order[order-1][ctx][token] = count`
  - add-α smoothing (default α=0.25)
  - optional backoff behaviour

### Scoring (optional)
`NGramModel.nll(seq)` and `.perplexity(seq)` allow quantitative evaluation of harmonic function modelling.

### Sampling
`NGramModel.sample(ctx, rng)` samples the next token from smoothed probabilities.

---

## Two-stage harmony model

### Configuration

`HarmonyConfig` controls:
- `num_bars`: phrase length (e.g., 8)
- `k_func`, `alpha_func`: n-gram order + smoothing for function model
- `enforce_final_cadence`: whether to force a cadence at the end
- `cadence_pattern`: default cadence pattern at half-bar resolution: `("D", "T")`

### Training

`train_harmony_model(pieces, cfg)`:

1. For each piece:
   - read quarter RN labels
   - compress to half-bar RN labels
   - (optional) simplify RN figures
   - map to function tokens (`T/PD/D`) and filter unknowns
2. Train `func_ngram` on function sequences
3. Fit `rn_by_func`:
   - empirical categorical distribution of RN figures conditioned on function

Returns a `HarmonyModel(cfg, func_ngram, rn_by_func)`.

### Sampling

`sample_function_plan(model)`:

- Samples `2 * num_bars` function tokens using the learned n-gram
- If cadence is enabled, the final two half-bars are forced to `D → T`

`realise_function_plan_to_rn(model, func_plan)`:

- For each function token, samples a Roman numeral from `P(RN|Function)`

`sample_harmony_plan(model)` returns both:
- function plan
- RN plan

---

## Design intent / rationale

- **Robustness:** Function classes reduce sensitivity to noisy RN labels.
- **Interpretability:** `T/PD/D` plans are easy to inspect and explain.
- **Cadence simplicity:** A cadence is enforced as a structural constraint (`D → T`), rather than learned implicitly.
- **Half-bar resolution:** 2 harmony tokens per bar captures more motion than bar-level harmony while staying manageable.

---

## Known limitations / TODOs

- Assumes 4/4 for half-bar grouping (2 beats). If meter varies, grouping should be time-signature aware.
- RN labels from `chordify()` can be noisy; simplification helps but does not eliminate all issues.
- Cadence pattern is fixed; could be conditioned on mode (major/minor) and/or section (V/F/E).
- `P(RN|Function)` ignores local context; a future extension could learn `P(RN_t | RN_{t-1}, Function_t)`.

