# Motif realisation module summary (`realise.py`)

This document summarises the design and behaviour of `src/motifgen/realise.py`.

## Goal

`realise.py` converts a **structural plan** (PCFG events) into a single **monophonic melody stream** by:

1. representing a user motif in a key-relative diatonic form  
2. applying motif transformation tokens (M0 / SEQ±n / INV / RET / REP / CAD)  
3. fitting each realised motif instance to the event’s allotted time  
4. placing all instances on a discrete time grid  
5. filling gaps (currently rests; later Markov infill)  
6. returning a `music21.stream.Part` for MIDI/score rendering  

`main.py` should call `realise_piece(...)` and handle I/O.

---

## Inputs

### User motif
Provided as a `music21.stream.Stream` (notes + rests). Templates can be hard-coded initially.

### PCFG plan
A list of `Event(tok, start_units, dur_units)` from `pcfg.py`, e.g.

- `Event(tok="M0", start_units=0, dur_units=4)`
- `Event(tok="SEQ+2", start_units=16, dur_units=4)`
- `Event(tok="CAD", start_units=56, dur_units=4)`

Time is measured on a discrete grid.

### Timing and key
- `units_per_beat` (default 2 → half-beat grid)
- `beats_per_bar` (default 4)
- `num_bars` sets total duration: `total_units = num_bars * beats_per_bar * units_per_beat`
- `key_obj` is a `music21.key.Key`

Optional: `rn_plan` is accepted for future harmony-aware constraints.

---

## Internal motif representation

Motifs are converted into diatonic events:

`MotifEvent(deg, oct, dur_units, is_rest)`

- `deg`: diatonic scale degree 0..6 relative to tonic  
- `oct`: octave (keeps register stable)  
- `dur_units`: duration on the time grid  
- `is_rest`: whether this event is a rest  

`Motif = List[MotifEvent]`

### Conversion helpers
- `motif_from_stream(stream, key_obj, units_per_beat) -> Motif`
- `motif_to_stream(motif, key_obj, units_per_beat, color=None) -> music21.Stream`

---

## Motif transformations (diatonic)

Transformations operate in scale-degree space.

- `M0` / `REP`: unchanged motif  
- `SEQ±n`: diatonic sequence via `diatonic_shift(motif, n)`  
- `INV`: diatonic inversion around an axis (default: first note degree)  
- `RET`: retrograde (reverse order)  
- `CAD`: handled via a special cadence template  

Entry point: `apply_motif_token(base_motif, tok) -> Motif`.

---

## Fitting motifs to event duration

Events provide a `dur_units` budget.

`fit_motif_to_duration(motif, target_units)`:
- scales `dur_units` proportionally  
- rounds to integers  
- adjusts final sum to exactly `target_units`  

This ensures each motif instance exactly fills its event window.

---

## Cadence handling

`CAD` is realised with `make_cadence_template(...)`, a simple cell ending on tonic:

- degree 1 → degree 0 (supertonic → tonic)  
- fitted to the event duration  

(More sophisticated cadence logic can be added later using the harmony plan.)

---

## Placement and assembly

### Collision policy (MVP)
If an event overlaps already-filled time, it is **skipped**.

An `occupied[total_units]` boolean array is used for overlap checks.

### Gaps
Unfilled time between motif instances is filled with **rests** (placeholder for Markov infill later).

---

## Main API

### `realise_piece(...) -> music21.stream.Part`

End-to-end method that:
1) converts motif stream → internal `Motif`  
2) sorts events  
3) realises + fits each motif token into a motif instance  
4) places instances onto the timeline  
5) assembles a `music21.stream.Part`  
6) colours motif instances (`note.style.color`) for visualisation  

The returned Part can be:
- written as MIDI (`.write("midi", ...)`)
- rendered in music21 (`.show()`)

---

## Design intent

- **Interpretability:** motif instances are explicit + highlightable  
- **Robustness:** diatonic operations reduce chromatic drift  
- **Maintainability:** placement/assembly lives in `realise.py`; `main.py` orchestrates  
- **Extensibility:** add harmony-aware constraints + Markov gap infill without changing PCFG interfaces  

