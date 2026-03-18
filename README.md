# MotifGen

MotifGen is a symbolic music generation system for short Baroque/early-Classical style pieces.
It combines:

- a probabilistic context-free grammar (PCFG) for phrase-level motif planning
- n-gram models for harmony generation and melodic infill

The system generates an 8-bar melody and accompaniment from either:

- a built-in demo motif, or
- a user-supplied MIDI or MusicXML motif file


## Repository layout

- `main.py`: main demo entrypoint used for generation
- `src/motifgen/`: package source code
- `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`: processed dataset splits
- `outputs/`: generated MIDI, score, and evaluation outputs


## Requirements

- Python `>= 3.8`
- the packages listed in `requirements.txt`

Optional:

- MuseScore, if you want `music21` to open rendered scores automatically with `score.show()`


## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pip install -e .` is not required for the coursework submission. The code is intended to be run from the repository root with `python main.py`.


## Run the generator

From the repository root:

```bash
python main.py
```

This uses the built-in demo motif and writes the generated MIDI to:

```text
outputs/midi/demo.mid
```


## Run with a motif file

You can provide a MIDI or MusicXML file as the motif input:

```bash
python main.py --motif-path path/to/motif.mid --mode major
```

or:

```bash
python main.py --motif-path path/to/motif.musicxml --mode minor
```

Arguments:

- `--motif-path`: optional path to a MIDI or MusicXML file to use as the motif
- `--mode`: generation mode, either `major` or `minor`


## Outputs

Running `main.py` will:

- load the training split from `data/train.jsonl`
- train the melody and harmony models
- generate one 8-bar piece
- write a MIDI file to `outputs/midi/demo.mid`
- attempt to open the score with `music21`
- print single-piece evaluation metrics to the terminal if evaluation succeeds


## System variants

The main entrypoint currently uses the full system:

```python
scfg = SystemConfig.full()
```

To demonstrate the ablation or baseline, edit `main.py` and replace that line with one of:

```python
scfg = SystemConfig.structure_only()
scfg = SystemConfig.melody_only_baseline()
```


## Notes

- The code expects the repository structure to remain unchanged.
- Run commands from the repository root so imports such as `src.motifgen` resolve correctly.
- If MuseScore is not installed, MIDI generation should still work even if score display does not open automatically.
