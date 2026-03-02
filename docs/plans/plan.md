One-week code plan (day-by-day, with deliverables)

Day 1 — Freeze representations + dataset pipeline (prevents chaos later)

Deliverables
	•	Decide tokenisation: (interval, duration) (plus bar boundaries) OR (scale_degree, duration) if you want fixed key.
	•	Build prepare_dataset.py:
	•	load music21 corpus subset (Bach chorales + Mozart/Vivaldi excerpts as you planned)
	•	extract monophonic line (keep it simple: top voice / highest pitch per timestep)
	•	save train/val/test token sequences + metadata
	•	Save a tiny “golden set” of ~20 extracted melodies for quick iteration.

Why this scores: rubric wants dataset description beyond naming it + clear pre-processing.  ￼

⸻

Day 2 — PCFG spec + sampler (addresses lecturer feedback directly)

Deliverables
	•	Write PCFG alphabet + rules in code (and in a README snippet for your report).
	•	Implement:
	•	sample_plan(num_bars, seed) → list of time-stamped plan events
e.g. (bar=0, token=M0), (bar=2, token=SEQ_UP_2), (bar=6, token=CADENCE)
	•	Add a “bar budget” guard so grammar recursion can’t blow up.

Minimum viable grammar
	•	S → V F E
	•	V → M0 M0 | M0
	•	F → F F | SEQ(+1) | SEQ(+2) | INV | M0
	•	E → CAD

Why this scores: Approach section demands implementational decisions + reproducibility; also your outline feedback explicitly asked for this detail.  ￼

⸻

Day 3 — Train k-gram Markov model + sampling API

Deliverables
	•	train_ngram.py outputs a pickle with counts + smoothing parameters.
	•	ngram.sample_next(context, constraints) supports constraints like:
	•	pitch range clamp
	•	prefer stepwise motion (optional)
	•	enforce cadence near end of phrase (light heuristic)
	•	Implement perplexity/NLL on validation (this becomes one of your quant metrics).

Why this scores: demonstrates “sequential model” technique and gives you a real quantitative evaluation method.  ￼

⸻

Day 4 — Realisation pipeline: plan → melody

Deliverables
	•	A single main.py that:
	1.	takes a motif (manual or hardcoded example)
	2.	samples PCFG plan
	3.	realises motif instances into notes
	4.	infills gaps with Markov sampling
	5.	outputs MIDI + score
	•	Generate 10–20 example outputs reliably (no crashes, no silent files).

Why this scores: the video needs to show code producing audible/visible outputs that clearly come from your techniques.  ￼

⸻

Day 5 — Motif detection + colour highlighting + baseline

Deliverables
	•	Motif matcher (transposition-invariant via interval-shape matching).
	•	Colour highlight motif instances in score render.
	•	Baseline system: Markov-only generation without PCFG plan + motif tokens.
	•	Same motif input, but no structure enforcement.

Why this scores: enables “comparison against baselines” and makes the system explainable (rubric loves this in Eval + Video).  ￼

⸻

Day 6 — Evaluation scripts + results table (small but meaningful)

Deliverables
Generate 30 samples per condition (baseline vs full).

Keep quantitative evaluation to 3 metrics that actually match your goals (avoid “random histogram soup”):
	1.	Motif coverage: # detected motif instances per piece; % bars containing an instance
	2.	Plan adherence: % planned motif tokens that appear in output (detectable)
	3.	Cadence/closure heuristic: phrase-final stability (e.g., last note in {1,3,5} and/or 2→1 or 7→1 motion near end)

Export a CSV + one small table for the report.

Why this scores: rubric explicitly wants at least one meaningful quantitative approach and results + reflection.  ￼

⸻

Day 7 — “Video-first” prep + report skeleton stubs

Deliverables
	•	Pick 2–3 best outputs + 1 failure case.
	•	Draft the 3-minute video script with timestamps (so you don’t ramble).
	•	Add report figure assets:
	•	one diagram of pipeline (PCFG → tokens → Markov infill → output)
	•	one small example with coloured motif highlights

Why this scores: video is 40 marks and needs planning, readable code excerpts, clear linkage between input and output.