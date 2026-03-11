
import random
import json
import music21 as m21
from src.motifgen import pcfg, realise, harmony_ngram, melody_ngram

def load_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def make_demo_motif() -> m21.stream.Stream:
    motif_stream = m21.stream.Stream()
    motif_stream.append(m21.note.Note("E4", quarterLength=0.25))
    motif_stream.append(m21.note.Note("D4", quarterLength=0.25))
    motif_stream.append(m21.note.Note("C4", quarterLength=0.5))
    motif_stream.append(m21.note.Note("D4", quarterLength=0.5))
    motif_stream.append(m21.note.Note("C4", quarterLength=0.5))
    #motif_stream.append(m21.note.Note("C4", quarterLength=0.25))
    #motif_stream.append(m21.note.Note("G3", quarterLength=0.5))
    #motif_stream.append(m21.note.Note("D4", quarterLength=1.0))
    return motif_stream


if __name__ == "__main__":
    # Musescore Setup
    us = m21.environment.UserSettings()
    us['musescoreDirectPNGPath'] = "/Applications/MuseScore 4.app"

    # 0) Config
    seed = 28 #74 #random.randint(0, 100) # 42
    num_bars = 4
    units_per_beat = 4
    beats_per_bar = 4
    total_units = num_bars * beats_per_bar * units_per_beat
    density = 0.80
    mode = "minor" # "minor"  # "major"

    print("Seed:", seed)
    print("Mode:", mode)
    # 1) Load dataset
    train_items = load_jsonl("data/train.jsonl")

    # 2) Sample phrase plan (PCFG events)
    grammar = pcfg.make_minimal_mvp_grammar(density=density)
    pcfg_cfg = pcfg.SamplerConfig(seed=seed, units_per_beat=units_per_beat, density=density, min_gap_units=0)
    events = grammar.sample_plan(num_bars=num_bars, cfg=pcfg_cfg, motif_dur_units=(8,))

    print("PCFG events:")
    for e in events:
        print(" ", e)

    # 3) Train melody n-gram (token-level)
    melody_seqs = [item["melody_tokens"] for item in train_items]
    mel_cfg = melody_ngram.NGramConfig(k=16, alpha=0.25, seed=seed)
    melody_model = melody_ngram.train_ngram(melody_seqs, mel_cfg)

    # 4) Train harmony model (function plan + RN plan at half-bar resolution)
    #harm_cfg = harmony_ngram.HarmonyConfig(num_bars=num_bars, k_func=6, alpha_func=0.25, seed=seed)
    #harm_model = harmony_ngram.train_harmony_model(train_items, harm_cfg)
    #func_plan, rn_plan = harmony_ngram.sample_harmony_plan(harm_model)

    # 4) Train 2 harmony models - 1 for major, 1 for minor
    major_cfg = harmony_ngram.HarmonyConfig(num_bars=num_bars, mode="major", seed=seed)
    minor_cfg = harmony_ngram.HarmonyConfig(num_bars=num_bars, mode="minor", seed=seed)

    harm_major = harmony_ngram.train_harmony_model(train_items, major_cfg)
    harm_minor = harmony_ngram.train_harmony_model(train_items, minor_cfg)

    harm_model = harm_minor if mode == "minor" else harm_major
    func_plan, rn_plan = harmony_ngram.sample_harmony_plan(harm_model)

    key_obj = m21.key.Key("a") if mode == "minor" else m21.key.Key("C")

    print("\nSampled function plan:", func_plan)
    print("Realised RN plan:", rn_plan)

    # 5) Choose key + motif template
    # (MVP: fixed key; later you can sample/condition this.)
    motif_stream = make_demo_motif()

    # 6) Build motif blocks as tokens for each event window
    base_motif = realise.motif_from_stream(motif_stream, key_obj=key_obj, units_per_beat=units_per_beat)
    motif_tokens_by_event = realise.motif_events_to_token_map(
        key_obj=key_obj,
        units_per_beat=units_per_beat,
        base_motif=base_motif,
        events=events,
        rn_plan=rn_plan,
        beats_per_bar=beats_per_bar,
    )

    # 7) Infill gaps with the melody n-gram
    infill_cfg = melody_ngram.InfillConfig(
        units_per_beat=units_per_beat,
        dur_set=(2.0, 1.0, 0.5),
        max_consecutive_rests=2,
        seed=seed,
    )

    color_map = {
        "M0": "#1f77b4",
        "REP": "#1f77b4",
        "SEQ+1": "#2ca02c",
        "SEQ+2": "#2ca02c",
        "SEQ-1": "#2ca02c",
        "SEQ-2": "#2ca02c",
        "INV": "#d62728",
        "RET": "#9467bd",
        "CAD": "#ff7f0e",
    }

    chord_pcs = melody_ngram.chord_pc_sets_from_rn_plan(rn_plan, key_obj=key_obj)
    melody_tokens, spans = melody_ngram.infill_timeline_with_spans(
        events=events,
        total_units=total_units,
        motif_tokens_by_event=motif_tokens_by_event,
        model=melody_model,
        cfg=infill_cfg,
        color_map=color_map,
        mode=mode,
        chord_pcs_by_halfbar=chord_pcs,
        rn_plan=rn_plan,
        key_obj=key_obj,
    )

    # 8) Realise melody tokens into a music21 part
    melody_part = realise.tokens_to_part(
        tokens=melody_tokens,
        key_obj=key_obj,
        default_octave=4,
        color_spans=spans,
    )
    melody_part.partName = "Melody"

    # 9) Realise harmony chords as a separate Part with RN labels
    harmony_part = realise.realise_harmony_part(
        key_obj=key_obj,
        rn_plan=rn_plan,
        num_bars=num_bars,
        units_per_beat=units_per_beat,
        beats_per_bar=beats_per_bar,
        bass_octave=3,
        part_name="Harmony",
    )

    # 10) Build score + output
    score = m21.stream.Score()
    score.insert(0.0, melody_part)
    score.insert(0.0, harmony_part)

    # Write MIDI
    out_midi = "outputs/midi/demo.mid"
    score.write("midi", fp=out_midi)
    print(f"\nWrote MIDI to: {out_midi}")

    # Show score (may open external viewer depending on environment)
    score.show()