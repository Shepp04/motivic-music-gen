# Goal of Pipeline
Produce token sequences like:
> BAR, NOTE(5, 0.5), NOTE(6, 0.5), NOTE(5, 1), BAR, NOTE(4, 0.25), ...

## Steps
1. Select corpus pieces
2. Extract a single monophonic line
3. Fix key + filter (transpose to Cmaj)
4. Quantize durations (nearest value in DUR_SET)
5. Insert bar boundaries
6. Save processed sequences + splits

## Files produced (in data/processed/)
* sequences_train.jsonl
* sequences_val.jsonl
* sequences_test.jsonl
* metadata.json (key, dur_set, corpus list, counts)

# What main.py should do
* load config
* load (or train) ngram
* sample PCFG plan
* realise motif tokens -> note tokens
* ngram infill
* render to MIDI + score