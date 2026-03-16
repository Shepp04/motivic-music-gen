Repo summary (structured, separate to report)

1) High-level intro
	•	Problem: Generate short symbolic pieces that feel structurally coherent in a Baroque/early-Classical idiom by explicitly modelling motivic development (repeat/transform a motif across a phrase), not just local note-to-note plausibility.
	•	Core idea: A top-down planner + bottom-up realiser:
	•	Plan where motif material happens and what transformations occur.
	•	Generate harmony in parallel.
	•	Realise motif instances, then infill the gaps with a melody n-gram under harmony/mode constraints.
	•	Outputs: Music21 score + MIDI with motif instances colour-highlighted; accompaniment rendered with selectable patterns.

⸻

2) What the system does (end-to-end)

Given a motif and a configuration (seed, length, density, mode):
	1.	PCFG phrase plan produces a list of scheduled events (token + start time + duration) across sections:
	•	Vordersatz (setup) → Fortspinnung (development) → Epilog/Cadence.
	2.	Harmony model samples a half-bar functional plan (T/PD/D) and realises it into a half-bar Roman numeral plan (RN).
	3.	Motif realisation creates concrete motif blocks for each PCFG event:
	•	Applies diatonic transforms (REP/INV/RET/SEQ±k/CAD).
	•	Enforces harmony-aware constraints (e.g., strong-beat chord tone alignment, minor-mode raised notes under V).
	4.	Melody infilling fills uncovered timeline gaps with a trained melody n-gram, with:
	•	mode constraints (diatonic),
	•	harmony constraints (strong beat chord-tone rate),
	•	optional rest gating,
	•	soft voice-leading preference.
	5.	Accompaniment rendering converts RN plan to a chordal part and renders a rhythmic accompaniment pattern (block chords / Alberti / arpeggiation), optionally varying by section.
	6.	Rendering and export:
	•	Score displayed via MuseScore, parts can be set to harpsichord.
	•	MIDI written to outputs/midi/.

⸻

3) Main components (by module)

src/motifgen/dataset.py
	•	Extracts training data from music21 corpus.
	•	Produces JSONL pieces containing:
	•	key (string),
	•	melody_tokens (with BAR tokens),
	•	harmony_tokens (quarter-note RN labels).
	•	Token format:
	•	Melody tokens: N:<pc>:<dur> and R:<dur> where pc is tonic-relative pitch class 0–11.
	•	Harmony: Roman numerals; later compressed to half-bar.

src/motifgen/pcfg.py
	•	Defines a PCFG over structure tokens (e.g., M0, REP, INV, RET, SEQUPk, SEQDNk, CAD).
	•	Two-stage sampling:
	1.	grammar expansion to terminals,
	2.	scheduler places them in time with density-controlled gaps.
	•	Supports longer pieces by spreading motif events throughout Fortspinnung and expanding sequence “runs” into multiple events.

src/motifgen/harmony_ngram.py
	•	Two-stage harmony generator:
	•	Function n-gram at half-bar resolution over {T, PD, D}.
	•	RN realiser: samples RN conditioned on function, with simplification/filtering to remove chromatic/modal-interchange noise.
	•	Major/minor separation supported (separate trained models), plus cadence constraints (final D→T; root position at cadence points).

src/motifgen/realise.py
	•	Converts motif stream → internal representation (diatonic degrees + durations).
	•	Applies motif transforms and fits blocks into scheduled windows.
	•	Harmony-aware motif adjustments:
	•	On strong onsets: prefer chord tones for the active RN.
	•	In minor over V: apply raised-note rules (7th always; 6th conditional per your final rule).
	•	Converts token sequences → music21 parts with:
	•	motif colour spans,
	•	register smoothing (allow nearby octaves to reduce large leaps).
	•	Realises RN plan to chords and labels them in the harmony/accompaniment part.

src/motifgen/melody_ngram.py
	•	Trains k-gram (Markov) model over melody tokens (BAR removed).
	•	Infills gaps between motif blocks on a unit grid:
	•	mode constraint: all notes diatonic to chosen mode (major/harmonic minor),
	•	harmony constraint: strong-beat chord-tone preference,
	•	voice-leading bias (soft penalty by pitch-class distance),
	•	optional rest toggle + minimum rest duration (avoid tiny rests causing syncopation).
	•	Returns:
	•	melody_tokens (flat sequence, no BAR),
	•	spans for colour-highlighting motif regions.

src/motifgen/accompaniment.py
	•	Renders RN plan into accompaniment with selectable patterns:
	•	block chords,
	•	Alberti bass (1–5–3–5),
	•	slow/fast arpeggiation.
	•	Can be constant for whole piece or vary by section boundaries (more active in Fortspinnung).

src/motifgen/eval.py
	•	Computes an evaluation suite per generated piece and across sets of pieces.
	•	Exports:
	•	Excel/CSV tables,
	•	plots (matplotlib).
	•	Designed for ablations and batch evaluation.

Entrypoints / orchestration
	•	main.py is currently a demo-style entrypoint that:
	•	trains models,
	•	generates one piece,
	•	calls evaluation on that piece vs corpus stats.
	•	(Recommended) src/motifgen/generate.py handles generation for many pieces and ablations; main.py stays thin.