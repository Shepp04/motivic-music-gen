# src/motifgen/generate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import random
from pathlib import Path

import music21 as m21

from . import pcfg, realise, harmony_ngram, melody_ngram, accompaniment


Token = str


# ----------------------------
# IO
# ----------------------------

def load_jsonl(path: str | Path) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_jsonl(items: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def event_to_dict(e: pcfg.Event) -> dict:
    return {"tok": e.tok, "start_units": e.start_units, "dur_units": e.dur_units}


def piece_to_jsonable(piece: dict) -> dict:
    """
    Make a generated piece JSONL-safe.
    - converts events (dataclasses) -> dict
    - drops non-serialisable objects if present
    """
    out = dict(piece)
    evs = out.get("events")
    if evs and isinstance(evs, list) and hasattr(evs[0], "tok"):
        out["events"] = [event_to_dict(e) for e in evs]
    return out


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class SystemConfig:
    """
    Toggleable components for ablations.
    """
    name: str = "full"

    # Harmony plan + chord constraints
    use_harmony: bool = True

    # Adjust motif strong beats to chord tones (and minor-V raising rules)
    use_harmony_aware_motif: bool = True

    # Constrain infill sampling using rn/chord tones, mode, raised-minor rules
    use_harmony_aware_infill: bool = True

    # Accompaniment rendering (block/alberti/arps)
    use_accompaniment: bool = True

    # Allow rests during infill
    allow_rests: bool = True


@dataclass(frozen=True)
class GenerateConfig:
    """
    Generation hyperparams (shared across systems).
    """
    num_bars: int = 16
    beats_per_bar: int = 4
    units_per_beat: int = 4

    # phrase plan density
    density: float = 0.64

    # mode selection for this run ("major" or "minor")
    mode: str = "major"

    # motif scheduling
    motif_dur_units: Tuple[int, ...] = (8,)  # 8 units at units_per_beat=4 => 2 beats
    motif_dur_probs: Optional[Tuple[float, ...]] = None

    # n-gram configs
    melody_k: int = 16
    melody_alpha: float = 0.25

    harmony_k_func: int = 6
    harmony_alpha_func: float = 0.25

    # infill config
    infill_dur_set: Tuple[float, ...] = (2.0, 1.0, 0.5)
    max_consecutive_rests: int = 2

    # accompaniment config
    accomp_units_per_beat: int = 2
    accomp_bass_octave: int = 2


@dataclass(frozen=True)
class Models:
    """
    Cached trained models so you don't retrain per-piece.
    """
    melody_model: melody_ngram.NGramModel
    harmony_major: harmony_ngram.HarmonyModel
    harmony_minor: harmony_ngram.HarmonyModel


# ----------------------------
# Training
# ----------------------------

def train_models(
    train_items: Sequence[dict],
    *,
    cfg: GenerateConfig,
    seed: int,
) -> Models:
    # Melody n-gram
    melody_seqs = [it["melody_tokens"] for it in train_items]
    mel_cfg = melody_ngram.NGramConfig(k=cfg.melody_k, alpha=cfg.melody_alpha, seed=seed)
    melody_model = melody_ngram.train_ngram(melody_seqs, mel_cfg)

    # Harmony models (train once each)
    major_cfg = harmony_ngram.HarmonyConfig(
        num_bars=cfg.num_bars,
        mode="major",
        seed=seed,
        k_func=cfg.harmony_k_func,
        alpha_func=cfg.harmony_alpha_func,
    )
    minor_cfg = harmony_ngram.HarmonyConfig(
        num_bars=cfg.num_bars,
        mode="minor",
        seed=seed,
        k_func=cfg.harmony_k_func,
        alpha_func=cfg.harmony_alpha_func,
    )

    harm_major = harmony_ngram.train_harmony_model(train_items, major_cfg)
    harm_minor = harmony_ngram.train_harmony_model(train_items, minor_cfg)

    return Models(melody_model=melody_model, harmony_major=harm_major, harmony_minor=harm_minor)


# ----------------------------
# Motif utilities
# ----------------------------

def make_demo_motif() -> m21.stream.Stream:
    """
    Replace this with user input later. Keep it here so batch generation is easy.
    """
    s = m21.stream.Stream()
    s.append(m21.note.Note("E4", quarterLength=0.25))
    s.append(m21.note.Note("F4", quarterLength=0.25))
    s.append(m21.note.Note("E4", quarterLength=0.25))
    s.append(m21.note.Note("D4", quarterLength=0.25))
    s.append(m21.note.Note("C4", quarterLength=1.0))
    return s


# ----------------------------
# Generation (single + batch)
# ----------------------------

def generate_one(
    *,
    models: Models,
    gcfg: GenerateConfig,
    scfg: SystemConfig,
    seed: int,
    motif_stream: Optional[m21.stream.Stream] = None,
) -> Tuple[dict, m21.stream.Score]:
    """
    Returns:
      - piece dict (for eval/jsonl)
      - music21 Score (for rendering / MIDI)
    """
    rng = random.Random(seed)

    num_bars = gcfg.num_bars
    upb = gcfg.units_per_beat
    bpb = gcfg.beats_per_bar
    total_units = num_bars * bpb * upb

    mode = gcfg.mode
    if mode not in ("major", "minor"):
        raise ValueError("GenerateConfig.mode must be 'major' or 'minor'")

    # Key is fixed
    key_obj = m21.key.Key("a") if mode == "minor" else m21.key.Key("C")

    # 1) Phrase plan
    grammar = pcfg.make_grammar(density=gcfg.density)
    pcfg_cfg = pcfg.SamplerConfig(
        seed=seed,
        units_per_beat=upb,
        beats_per_bar=bpb,
        density=gcfg.density,
        min_gap_units=0,
    )

    events = grammar.sample_plan(
        num_bars=num_bars,
        cfg=pcfg_cfg,
        motif_dur_units=gcfg.motif_dur_units,
        motif_dur_probs=gcfg.motif_dur_probs,
    )

    # 2) Harmony plan (or disable)
    if scfg.use_harmony:
        harm_model = models.harmony_minor if mode == "minor" else models.harmony_major
        func_plan, rn_plan = harmony_ngram.sample_harmony_plan(harm_model)
    else:
        func_plan = []
        rn_plan = ["N"] * (2 * num_bars)

    # 3) Motif blocks
    if motif_stream is None:
        motif_stream = make_demo_motif()

    base_motif = realise.motif_from_stream(motif_stream, key_obj=key_obj, units_per_beat=upb)

    motif_tokens_by_event = realise.motif_events_to_token_map(
        key_obj=key_obj,
        units_per_beat=upb,
        base_motif=base_motif,
        events=events,
        rn_plan=rn_plan if scfg.use_harmony_aware_motif and scfg.use_harmony else None,
        beats_per_bar=bpb,
    )

    # 4) Infill gaps (melody n-gram)
    infill_cfg = melody_ngram.InfillConfig(
        units_per_beat=upb,
        dur_set=gcfg.infill_dur_set,
        allow_rests=scfg.allow_rests,
        max_consecutive_rests=gcfg.max_consecutive_rests,
        seed=seed,
    )

    color_map = {
        "M0": "#1f77b4",
        "REP": "#1f77b4",
        "INV": "#d62728",
        "RET": "#9467bd",
        "CAD": "#ff7f0e",
        # sequences (if present)
        "SEQ+1": "#2ca02c",
        "SEQ+2": "#2ca02c",
        "SEQ+3": "#2ca02c",
        "SEQ+4": "#2ca02c",
        "SEQ-1": "#04d9ff",
        "SEQ-2": "#04d9ff",
        "SEQ-3": "#04d9ff",
        "SEQ-4": "#04d9ff",
    }

    chord_pcs = None
    rn_for_infill = None
    if scfg.use_harmony and scfg.use_harmony_aware_infill:
        chord_pcs = melody_ngram.chord_pc_sets_from_rn_plan(rn_plan, key_obj=key_obj)
        rn_for_infill = rn_plan

    melody_tokens, spans = melody_ngram.infill_timeline_with_spans(
        events=events,
        total_units=total_units,
        motif_tokens_by_event=motif_tokens_by_event,
        model=models.melody_model,
        cfg=infill_cfg,
        color_map=color_map,
        mode=mode,
        chord_pcs_by_halfbar=chord_pcs,
        rn_plan=rn_for_infill,
        key_obj=key_obj,
    )

    # 5) Render parts
    melody_part = realise.tokens_to_part(
        tokens=melody_tokens,
        key_obj=key_obj,
        default_octave=5,
        color_spans=spans,
    )
    melody_part.partName = "Melody"

    score = m21.stream.Score()
    score.insert(0.0, melody_part)

    if scfg.use_accompaniment and scfg.use_harmony:
        section = accompaniment.SectionSpec(v_bars=2, f_bars=num_bars - 3, e_bars=1)
        style_plan = accompaniment.sample_style_plan(
            num_bars=num_bars,
            section=section,
            mode="by_section",
            seed=seed,
        )
        acc_part = accompaniment.realise_accompaniment_part(
            key_obj=key_obj,
            rn_plan=rn_plan,
            style_plan=style_plan,
            cfg=accompaniment.AccompConfig(
                units_per_beat=gcfg.accomp_units_per_beat,
                beats_per_bar=bpb,
                bass_octave=gcfg.accomp_bass_octave,
            ),
        )
        acc_part.partName = "Accompaniment"
        score.insert(0.0, acc_part)

    # instruments (optional convenience)
    for p in score.parts:
        try:
            p.insert(0.0, m21.instrument.Harpsichord())
        except Exception:
            pass

    piece = {
        "seed": seed,
        "system": scfg.name,
        "mode": mode,
        "num_bars": num_bars,
        "units_per_beat": upb,
        "beats_per_bar": bpb,
        "density": gcfg.density,
        "melody_tokens": melody_tokens,
        "rn_plan": rn_plan,
        "func_plan": func_plan,
        "events": events,
    }

    return piece, score


def generate_many(
    *,
    models: Models,
    gcfg: GenerateConfig,
    scfg: SystemConfig,
    seeds: Sequence[int],
    motif_stream: Optional[m21.stream.Stream] = None,
) -> List[dict]:
    pieces: List[dict] = []
    for seed in seeds:
        piece, _ = generate_one(models=models, gcfg=gcfg, scfg=scfg, seed=seed, motif_stream=motif_stream)
        pieces.append(piece)
    return pieces