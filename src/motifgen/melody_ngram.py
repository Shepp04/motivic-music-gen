# src/motifgen/melody_ngram.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random
import re

import music21 as m21
from motifgen.pcfg import Event

Token = str
Context = Tuple[Token, ...]

_NOTE_RE = re.compile(r"^N:(\d+):([0-9.]+)$")
_REST_RE = re.compile(r"^R:([0-9.]+)$")
_RN_ROOT_RE = re.compile(r"^([ivIV]+o?)")

# ----------------------------
# Mode constraints
# ----------------------------

Mode = str  # "major" | "minor"

MAJOR_PCS = {0, 2, 4, 5, 7, 9, 11}
HARM_MINOR_PCS = {0, 2, 3, 5, 7, 8, 10, 11}  # harmonic minor (raised 7 allowed)


def allowed_note_pcs(mode: Mode) -> set[int]:
    return set(MAJOR_PCS) if mode == "major" else set(HARM_MINOR_PCS)


# ----------------------------
# Token utilities
# ----------------------------

def is_note(tok: str) -> bool:
    return _NOTE_RE.match(tok) is not None


def is_rest(tok: str) -> bool:
    return _REST_RE.match(tok) is not None


def _rn_root_is_V(rn: str) -> bool:
    if not rn or rn == "N":
        return False
    s = rn.strip().replace("°", "o").replace("ø", "o").split("/")[0]
    m = _RN_ROOT_RE.match(s)
    return bool(m and m.group(1) == "V")

def pc_of(tok: str) -> Optional[int]:
    m = _NOTE_RE.match(tok)
    return int(m.group(1)) if m else None


def token_dur_units(tok: str, units_per_beat: int) -> int:
    m = _NOTE_RE.match(tok)
    if m:
        return max(1, int(round(float(m.group(2)) * units_per_beat)))
    m = _REST_RE.match(tok)
    if m:
        return max(1, int(round(float(m.group(1)) * units_per_beat)))
    return 0


def last_note_pc(tokens: Sequence[str]) -> Optional[int]:
    for t in reversed(tokens):
        if is_note(t):
            return pc_of(t)
    return None


def pc_dist(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


# ----------------------------
# Chord tone sets from RN plan
# ----------------------------

def chord_pc_sets_from_rn_plan(rn_plan: Sequence[str], *, key_obj: m21.key.Key) -> List[set[int]]:
    tonic_pc = key_obj.tonic.pitchClass
    out: List[set[int]] = []
    for rn in rn_plan:
        rn = (rn or "N").strip()
        if rn == "N":
            out.append(set())
            continue
        rn_norm = rn.replace("°", "o").replace("ø", "o")
        try:
            rn_obj = m21.roman.RomanNumeral(rn_norm, key_obj)
            pcs_rel = {(pc - tonic_pc) % 12 for pc in rn_obj.pitchClasses}
            out.append(pcs_rel)
        except Exception:
            out.append(set())
    return out


def halfbar_index_for_unit(cur_unit: int, *, units_per_beat: int) -> int:
    return cur_unit // (2 * units_per_beat)


def is_strong_beat(
    cur_unit: int,
    *,
    units_per_beat: int,
    beats_per_bar: int,
    strong_beats: Tuple[int, ...],
) -> bool:
    beat = (cur_unit // units_per_beat) % beats_per_bar
    return beat in strong_beats


# ----------------------------
# N-gram model
# ----------------------------

@dataclass
class NGramConfig:
    k: int = 4
    alpha: float = 0.25
    seed: Optional[int] = None


@dataclass
class NGramModel:
    k: int
    alpha: float
    vocab: List[Token]
    counts_by_order: List[Dict[Context, Dict[Token, int]]]

    def prob(self, tok: Token, ctx: Context) -> float:
        order = min(self.k, len(ctx) + 1)
        ctx = ctx[-(order - 1):] if order > 1 else ()
        counts = self.counts_by_order[order - 1].get(ctx)

        if counts is None and order > 1:
            return self.prob(tok, ctx[1:])

        counts = counts or {}
        total = sum(counts.values())
        c = counts.get(tok, 0)
        V = len(self.vocab)
        return (c + self.alpha) / (total + self.alpha * V)

    def sample_weighted(self, ctx: Context, rng: random.Random, subset: Sequence[Token], prev_pc: Optional[int], lam: float) -> Token:
        # weight = P_ngram(tok|ctx) * exp(-lam * pc_dist(prev, tok_pc))
        weights: List[float] = []
        for t in subset:
            w = self.prob(t, ctx)
            if lam > 1e-9 and prev_pc is not None and is_note(t):
                pc = pc_of(t)
                if pc is not None:
                    w *= math.exp(-lam * pc_dist(prev_pc, pc))
            weights.append(w)

        s = sum(weights)
        if s <= 0:
            return rng.choice(list(subset))
        x = rng.random() * s
        acc = 0.0
        for t, w in zip(subset, weights):
            acc += w
            if x <= acc:
                return t
        return subset[-1]


def train_ngram(seqs: Iterable[Sequence[Token]], cfg: NGramConfig) -> NGramModel:
    seqs_list: List[List[Token]] = []
    vocab_set = set()
    for s in seqs:
        s2 = [t for t in s if t != "BAR"]
        if not s2:
            continue
        seqs_list.append(s2)
        vocab_set.update(s2)

    vocab = sorted(vocab_set)
    if not vocab:
        raise ValueError("train_ngram: empty vocab after filtering")

    counts_by_order: List[Dict[Context, Dict[Token, int]]] = [dict() for _ in range(cfg.k)]

    def bump(order: int, ctx: Context, tok: Token) -> None:
        d = counts_by_order[order - 1].setdefault(ctx, {})
        d[tok] = d.get(tok, 0) + 1

    for seq in seqs_list:
        for i, tok in enumerate(seq):
            for order in range(1, cfg.k + 1):
                ctx_len = order - 1
                if ctx_len == 0:
                    ctx = ()
                else:
                    start = i - ctx_len
                    if start < 0:
                        continue
                    ctx = tuple(seq[start:i])
                bump(order, ctx, tok)

    return NGramModel(k=cfg.k, alpha=cfg.alpha, vocab=vocab, counts_by_order=counts_by_order)


# ----------------------------
# Infill
# ----------------------------

@dataclass
class InfillConfig:
    units_per_beat: int = 2
    dur_set: Tuple[float, ...] = (2.0, 1.0, 0.5, 0.25)
    max_consecutive_rests: int = 2
    seed: Optional[int] = None

    # Voice leading (soft)
    voice_leading_lambda: float = 0.9

    # Strong-beat harmony behaviour
    strong_beats: Tuple[int, ...] = (0, 2)  # beats 1 and 3 in 4/4
    beats_per_bar: int = 4
    strong_beat_pick_closest: bool = False  # if True, deterministically choose closest chord tone


def _fits_duration(tok: Token, remaining_units: int, cfg: InfillConfig) -> bool:
    du = token_dur_units(tok, cfg.units_per_beat)
    if du <= 0 or du > remaining_units:
        return False

    # require duration to be in dur_set
    m = _NOTE_RE.match(tok)
    if m and float(m.group(2)) not in cfg.dur_set:
        return False
    m = _REST_RE.match(tok)
    if m and float(m.group(1)) not in cfg.dur_set:
        return False

    return True


def _mode_filter(subset: Sequence[Token], allowed_pcs: set[int]) -> List[Token]:
    out: List[Token] = []
    for t in subset:
        if is_note(t):
            pc = pc_of(t)
            if pc is not None and pc in allowed_pcs:
                out.append(t)
        else:
            out.append(t)  # rests always allowed
    return out


def _strong_beat_chord_filter(subset: Sequence[Token], chord_pcs: set[int]) -> List[Token]:
    if not chord_pcs:
        return list(subset)
    notes = [t for t in subset if is_note(t) and (pc_of(t) in chord_pcs)]
    others = [t for t in subset if not is_note(t)]
    return notes + others if notes else list(subset)


def _substitute_pc_token(tok: Token, new_pc: int) -> Optional[Token]:
    """
    If tok is NOTE token N:<pc>:<dur>, return same dur with new_pc.
    """
    m = _NOTE_RE.match(tok)
    if not m:
        return None
    dur = m.group(2)
    return f"N:{new_pc}:{dur}"


def _apply_raised_minor_over_V(
    subset: List[Token],
    *,
    rn_plan: Sequence[str],
    key_obj: m21.key.Key,
    cur_unit: int,
    units_per_beat: int,
) -> List[Token]:
    """
    In minor key, over V harmony, forbid pcs 8 and 10 (b6, b7) and instead allow 9, 11.
    Works by filtering candidates; actual substitution is handled after sampling too.
    """
    if (key_obj.mode or "").lower() != "minor":
        return subset

    hi = halfbar_index_for_unit(cur_unit, units_per_beat=units_per_beat)
    rn = rn_plan[hi] if 0 <= hi < len(rn_plan) else "N"
    if not _rn_root_is_V(rn):
        return subset

    forbidden = {8, 10}
    # Remove forbidden NOTE tokens under V
    out = []
    for t in subset:
        if is_note(t):
            pc = pc_of(t)
            if pc in forbidden:
                continue
        out.append(t)

    return out if out else subset  # don’t delete everything accidentally


def fill_gap_tokens(
    *,
    model: NGramModel,
    context: List[Token],
    gap_units: int,
    gap_start_unit: int,
    cfg: InfillConfig,
    rng: random.Random,
    mode: Mode,
    chord_pcs_by_halfbar: Optional[Sequence[set[int]]] = None,
    rn_plan: Optional[Sequence[str]] = None,
    key_obj: Optional[m21.key.Key] = None,
) -> List[Token]:
    """
    Fill a gap with tokens summing to gap_units.

    Constraints:
      - ALL generated NOTE tokens must be diatonic to `mode`
      - On strong beats, optionally constrain notes to chord tones (if chord_pcs_by_halfbar provided)
      - Soft voice-leading preference to stay near previous note pc
    """
    allowed_pcs = allowed_note_pcs(mode)

    vocab_set = set(model.vocab)
    if key_obj is None:
        # fallback: construct from mode + C tonic
        key_obj = m21.key.Key("C", "minor" if mode == "minor" else "major")

    out: List[Token] = []
    remaining = gap_units
    rest_run = 0
    cur_unit = gap_start_unit

    while remaining > 0:
        ctx = tuple(context[-(model.k - 1):]) if model.k > 1 else ()
        prev_pc = last_note_pc(context)

        subset = [t for t in model.vocab if _fits_duration(t, remaining, cfg)]
        if not subset:
            # fallback: smallest rest
            ql = min(cfg.dur_set)
            du = max(1, int(round(ql * cfg.units_per_beat)))
            tok = f"R:{ql}"
            out.append(tok)
            context.append(tok)
            remaining -= du
            rest_run += 1
            cur_unit += du
            continue

        # Avoid too many consecutive rests
        if rest_run >= cfg.max_consecutive_rests:
            nonrests = [t for t in subset if not is_rest(t)]
            subset = nonrests or subset

        # mode constraint on ALL notes
        subset = _mode_filter(subset, allowed_pcs)
        if not subset:
            # if mode filtering killed everything, allow rests only
            subset = [t for t in model.vocab if is_rest(t) and _fits_duration(t, remaining, cfg)]
            if not subset:
                subset = [f"R:{min(cfg.dur_set)}"]

        if rn_plan is not None:
            subset = _apply_raised_minor_over_V(
                subset,
                rn_plan=rn_plan,
                key_obj=key_obj,
                cur_unit=cur_unit,
                units_per_beat=cfg.units_per_beat,
            )

        # Strong-beat chord constraint (optional)
        if chord_pcs_by_halfbar is not None and is_strong_beat(
            cur_unit,
            units_per_beat=cfg.units_per_beat,
            beats_per_bar=cfg.beats_per_bar,
            strong_beats=cfg.strong_beats,
        ):
            hi = halfbar_index_for_unit(cur_unit, units_per_beat=cfg.units_per_beat)
            chord_pcs = chord_pcs_by_halfbar[hi] if 0 <= hi < len(chord_pcs_by_halfbar) else set()
            subset = _strong_beat_chord_filter(subset, chord_pcs)

            # Optional: pick closest chord tone deterministically
            if cfg.strong_beat_pick_closest and prev_pc is not None and chord_pcs:
                chord_notes = [t for t in subset if is_note(t) and (pc_of(t) in chord_pcs)]
                if chord_notes:
                    best = min(
                        chord_notes,
                        key=lambda t: (pc_dist(prev_pc, pc_of(t)), -model.prob(t, ctx)),
                    )
                    tok = best
                else:
                    tok = model.sample_weighted(ctx, rng, subset, prev_pc, cfg.voice_leading_lambda)
            else:
                tok = model.sample_weighted(ctx, rng, subset, prev_pc, cfg.voice_leading_lambda)
        else:
            tok = model.sample_weighted(ctx, rng, subset, prev_pc, cfg.voice_leading_lambda)

        du = token_dur_units(tok, cfg.units_per_beat)
        if du <= 0 or du > remaining:
            continue

        out.append(tok)
        context.append(tok)
        remaining -= du
        cur_unit += du
        rest_run = rest_run + 1 if is_rest(tok) else 0

    return out


def infill_timeline_with_spans(
    *,
    events: Sequence[Event],
    total_units: int,
    motif_tokens_by_event: Dict[Tuple[int, int], List[Token]],
    model: NGramModel,
    cfg: InfillConfig,
    color_map: Dict[str, str],
    mode: Mode,
    chord_pcs_by_halfbar: Optional[Sequence[set[int]]] = None,
    rn_plan: Optional[Sequence[str]] = None,
    key_obj: Optional[m21.key.Key] = None,
) -> Tuple[List[Token], List[Tuple[int, int, str]]]:
    rng = random.Random(cfg.seed)
    evs = sorted(events, key=lambda e: e.start_units)

    out: List[Token] = []
    context: List[Token] = []
    spans: List[Tuple[int, int, str]] = []
    cur = 0

    for e in evs:
        s = max(0, e.start_units)
        e_end = min(total_units, e.start_units + e.dur_units)

        if s > cur:
            out.extend(
                fill_gap_tokens(
                    model=model,
                    context=context,
                    gap_units=s - cur,
                    gap_start_unit=cur,
                    cfg=cfg,
                    rng=rng,
                    mode=mode,
                    chord_pcs_by_halfbar=chord_pcs_by_halfbar,
                    rn_plan=rn_plan,
                    key_obj=key_obj,
                )
            )
            cur = s

        block = motif_tokens_by_event.get((e.start_units, e.dur_units), [])
        if not block:
            # deterministic rest fill
            remaining = e_end - cur
            while remaining > 0:
                ql = min(cfg.dur_set)
                du = max(1, int(round(ql * cfg.units_per_beat)))
                if du > remaining:
                    ql = max(d for d in cfg.dur_set if int(round(d * cfg.units_per_beat)) <= remaining)
                    du = int(round(ql * cfg.units_per_beat))
                block.append(f"R:{ql}")
                remaining -= du

        a = len(out)
        out.extend(block)
        b = len(out)

        col = color_map.get(e.tok)
        if col and b > a:
            spans.append((a, b, col))

        context.extend(block)
        cur = e_end

    if cur < total_units:
        out.extend(
            fill_gap_tokens(
                model=model,
                context=context,
                gap_units=total_units - cur,
                gap_start_unit=cur,
                cfg=cfg,
                rng=rng,
                mode=mode,
                chord_pcs_by_halfbar=chord_pcs_by_halfbar,
            )
        )

    return out, spans