A. The metrics to lock in (with what each proves)

1) Strong-beat chord-tone rate (SBCTR)

What it measures: harmony–melody consonance, i.e., whether your harmony-aware constraints actually worked.
How: for each melody onset that lands on a strong beat (your definition: first subdivision of beats 1 & 3), check if its pc ∈ chord tone pcs for the active RN half-bar.
Report: mean ± std across N pieces.

Ablation sensitivity:
	•	drops sharply when you turn off harmony-aware motif adjustment and harmony-aware infill.

⸻

2) Pitch interval statistics vs corpus

What it measures: melodic “singability” / style plausibility (Bach-ish lines have lots of steps, few huge leaps).
How: convert consecutive notes to semitone intervals (or diatonic steps if you prefer), then compute:
	•	% stepwise (|interval| ≤ 2)
	•	% leaps ≥ 7 semitones
	•	mean |interval|, 95th percentile |interval|

Compute the same on a held-out corpus slice (same tokenisation) and report distance to corpus (e.g., absolute difference per statistic).

Ablation sensitivity:
	•	gets worse if you remove voice-leading weighting, allow small rests, or remove mode filtering.

⸻

3) Self-similarity matrix “structure score” (SSM-Score)

What it measures: long-term structure / repetition (your main thesis: PCFG + motif tokens gives coherent repetition).
How (simple and defensible):
	•	segment the melody into bars (or half-bars).
	•	represent each segment with a small feature vector, e.g.:
	•	pitch-class histogram (12-d)
	•	onset/rhythm histogram (grid-based)
	•	optionally interval bigram histogram
	•	compute cosine similarity between all segment vectors → SSM (B×B).
	•	compute a scalar score like:
	•	mean similarity of off-diagonal entries (excluding near-diagonal), and/or
	•	diagonal stripe score: average similarity for lag 2,4,8 bars (periodic repetition), minus shuffled baseline.

This avoids “hand-wavy”: you can show the SSM heatmaps for each system.

Ablation sensitivity:
	•	biggest winner: PCFG/motif system should show clearer stripes than “ngram-only”.

⸻

4) Motif placement metrics (because you explicitly model motifs)

These are almost free because you already have events and motif spans.

4a) Motif coverage
% of total time in motif events (vs infill). Should increase with density knob.

4b) Motif dispersion / clumping
Compute motif instances per bar and report variance (or Gini).
Goal: motifs are distributed throughout Fortspinnung rather than clustered.

4c) Motif fidelity (optional but strong)
For each motif instance, compare its pc sequence to the base motif under the intended transform:
	•	edit distance / normalized mismatch rate

Ablation sensitivity:
	•	distinguishes “structure token plan” vs “just a few motifs then ngram drifts”.

---

C. Ablation suite (minimal but convincing)

You want one baseline + two ablations that line up with the metrics:
	1.	Baseline (B0): Melody n-gram only (no PCFG events, no motif injection, no harmony conditioning).
	2.	Ablation (A1): PCFG + motif tokens but no harmony-aware adjustments/infill (structure-only).
	3.	Full (F): PCFG + motif + harmony-aware motif adjustment + harmony-aware infill + voice-leading.

Optional 4th if you have time:
4) A2: Full but no voice-leading weighting (tests interval stats strongly).

Run each for N = 30–50 pieces with fixed seeds. That’s enough to avoid “small number of outputs” (and you can justify time constraints).

⸻

D. What eval.py should output (so your report writes itself)

For each system variant:
	•	a CSV/JSON with per-piece metrics
	•	summary table (mean ± std)
	•	3–5 representative SSM heatmaps (one per system) + captions
	•	one histogram plot for interval magnitudes (system vs corpus)

This directly addresses the “figures without explanation” weakness: your captions can say “PCFG produces higher lag-4 similarity stripes than baseline”.

---

Final locked list (if you want the cleanest set)

If you want the tightest set that covers everything:
	1.	SB chord-tone rate
	2.	Interval statistics vs corpus (stepwise %, leap%, mean |interval|)
	3.	SSM score (bar-level) + include SSM figure
	4.	Motif coverage + motif dispersion (variance per bar)
	5.	Cadence correctness/alignment pass rate
