# Dataset pipeline (music21 → tokens)

This document summarises the current `dataset.py` pipeline used to build training/validation/test data for the project.

## Purpose

Given a Bach chorale (music21 corpus), we extract:

- **Melody token stream** (monophonic, bar-aligned)
- **Harmony labels** (Roman numerals at quarter-note resolution)

Each processed item is saved as a JSON object in a JSONL file.

## Corpus selection

`load_corpus_ids(limit)` uses:

- `music21.corpus.chorales.Iterator(returnType="filename")`

It returns parseable corpus identifiers like `bach/bwv269`, which are passed to `music21.corpus.parse()`.

## Per-piece processing

`process_piece(corpus_id)`:

1. Parses the score: `score = m21.corpus.parse(corpus_id)`
2. Analyses the **original key** once (fallback: C major):
   - `key_obj = score.analyze("key")`
3. Extracts a melody part via `extract_melody(score)`
4. Tokenises melody via `tokenise_melody(melody_part, key_obj)`
5. Extracts harmony labels via `extract_harmony(score, key_obj)`
6. Returns a dict:

```json
{
  "id": "bach/bwvXXX",
  "key": "C major",
  "melody_tokens": ["BAR", "N:0:1.0", ...],
  "harmony_tokens": ["I", "V6", "N", ...]
}
```

## Melody extraction

`extract_melody(score)`:

- If the score has 4+ parts (typical chorales), returns **part 0** (assumed soprano).
- Otherwise selects the part with the **highest average MIDI pitch**.

## Melody representation

Melody tokens are strings:

- `BAR` — bar boundary
- `N:<pc>:<dur>` — note token  
  - `pc` is **tonic-relative pitch class** in `[0..11]`  
    - computed as `(pitchClass - tonic.pitchClass) % 12`
  - `dur` is in quarterLength units
- `R:<dur>` — rest token (same duration units)

### Duration quantisation

The allowed duration set (MVP) is:

- `{2.0, 1.0, 0.5}` quarterLength  
  (minim, crotchet, quaver)

Any note/rest duration is snapped to the nearest value with `_snap_duration()`.

### Bar alignment / padding

For each measure:

- Determine bar length from the measure time signature:
  - `bar_len = ts.barDuration.quarterLength` (fallback 4.0)
- Emit `BAR`
- Consume `notesAndRests` in order
- Split events if they would exceed the bar
- Pad the bar with rests so each bar sums to `bar_len`

### Anacrusis (pickup)

`get_anacrusis_shift()` attempts to detect an incomplete first measure and returns its duration (quarterLength).  
If `shift > 0`, `tokenise_melody()` skips **measure index 0**.

> Note: Some corpus encodings make pickup detection unreliable (pickup measures can appear “full-length”).  
> Current approach may leave slight BAR misalignment for those cases.

## Harmony extraction

`extract_harmony(score, key_obj)`:

1. `chordified = score.chordify()`
2. Compute pickup shift: `shift = get_anacrusis_shift(score)`
3. Sample harmony at **crotchet (quarter-note)** resolution:

- For `i = 0..total_q-1`, query chordified score at time `t = shift + i`
- Get chord(s) sounding in `[t, t+1)`
- Compute a Roman numeral relative to `key_obj`:

  - `rn = romanNumeralFromChord(chord, key_obj)`
  - store `rn.figure`

If no chord is found or RN fails, store `"N"`.

## Splitting + saving

`train_val_test_split(items, seed)`:

- deterministic shuffle
- 80% train, 10% val, 10% test

`save_jsonl(items, path)` writes one JSON dict per line.

## Example usage

The module’s `__main__` builds the dataset:

- `corpus_ids = load_corpus_ids(limit=100)`
- `data = [process_piece(cid) for cid in corpus_ids]`
- split + save to:
  - `data/train.jsonl`
  - `data/val.jsonl`
  - `data/test.jsonl`

## Known limitations / TODOs

- **Pickup detection** can fail for some encodings; may require a more robust barline-based alignment later.
- Melody extraction assumes part 0 is soprano for chorales; verify if using other corpora.
- Roman numeral labels from `chordify()` can be noisy (non-chord tones); consider per-bar labels if needed.
