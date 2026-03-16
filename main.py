# main.py
from __future__ import annotations

import random
import music21 as m21

from src.motifgen import eval as eval_mod
from src.motifgen.generate import (
    GenerateConfig,
    SystemConfig,
    load_jsonl,
    train_models,
    generate_one,
)

if __name__ == "__main__":
    # MuseScore 4 (optional)
    try:
        us = m21.environment.UserSettings()
        us["musescoreDirectPNGPath"] = "/Applications/MuseScore 4.app"
    except Exception:
        pass

    # ---- Config ----
    seed = random.randint(0, 10_000)
    gcfg = GenerateConfig(
        num_bars=8,
        units_per_beat=4,
        beats_per_bar=4,
        density=0.50,
        mode="major",              # "minor"
        motif_dur_units=(16,),      # keep fixed to avoid motif augmentation
        infill_dur_set=(2.0, 1.0, 0.5),
    )
    # Swap this for SystemConfig.structure_only() or
    # SystemConfig.melody_only_baseline() when generating ablations.
    scfg = SystemConfig.full()
    
    print("Seed:", seed)
    print("Mode:", gcfg.mode)
    print("System:", scfg.name)

    # ---- Load dataset + train once ----
    train_items = load_jsonl("data/train.jsonl")
    models = train_models(train_items, cfg=gcfg, seed=seed)

    # ---- Generate ----
    piece, score = generate_one(models=models, gcfg=gcfg, scfg=scfg, seed=seed)

    # Write MIDI
    out_midi = "outputs/midi/demo.mid"
    score.write("midi", fp=out_midi)
    print("Wrote MIDI to:", out_midi)

    # Show score
    score.show()

    # ---- Optional: evaluate single piece (uses your existing eval.py API) ----
    try:
        eval_cfg = eval_mod.EvalConfig(units_per_beat=gcfg.units_per_beat, beats_per_bar=gcfg.beats_per_bar)
        val_items = load_jsonl("data/val.jsonl")
        test_items = load_jsonl("data/test.jsonl")
        corpus = eval_mod.compute_corpus_stats(val_items + test_items, cfg=eval_cfg)
        metrics, _ssm = eval_mod.evaluate_piece(piece, cfg=eval_cfg, corpus=corpus)
        print("\nEvaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print("\n[Eval skipped due to error]:", e)
