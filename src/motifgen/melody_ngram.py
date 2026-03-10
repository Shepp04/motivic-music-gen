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


# ----------------------------
# Token parsing helpers
# ----------------------------

_NOTE_RE = re.compile(r"^N:(\d+):([0-9.]+)$")
_REST_RE = re.compile(r"^R:([0-9.]+)$")


def token_dur_units(tok: str, units_per_beat: int) -> int:
    """
    Convert token duration (quarterLength) to grid units.
    quarterLength 1.0 == 1 beat.
    """
    m = _NOTE_RE.match(tok)
    if m:
        ql = float(m.group(2))
        return max(1, int(round(ql * units_per_beat)))
    m = _REST_RE.match(tok)
    if m:
        ql = float(m.group(1))
        return max(1, int(round(ql * units_per_beat)))
    # unknown token: treat as 0 units (shouldn't happen)
    return 0


def is_note(tok: str) -> bool:
    return _NOTE_RE.match(tok) is not None


def is_rest(tok: str) -> bool:
    return _REST_RE.match(tok) is not None


def pc_of(tok: str) -> Optional[int]:
    m = _NOTE_RE.match(tok)
    return int(m.group(1)) if m else None


def pc_circular_dist(a: int, b: int) -> int:
    """Circular pitch-class distance on mod-12."""
    d = abs(a - b) % 12
    return min(d, 12 - d)

def last_note_pc(tokens: Sequence[str]) -> Optional[int]:
    """Find the most recent NOTE token pc in a token list."""
    for t in reversed(tokens):
        if is_note(t):
            return pc_of(t)
    return None


def make_note(pc: int, ql: float) -> str:
    return f"N:{pc}:{ql}"


def make_rest(ql: float) -> str:
    return f"R:{ql}"


def chord_pc_sets_from_rn_plan(
    rn_plan: Sequence[str],
    *,
    key_obj: m21.key.Key,
) -> List[set[int]]:
    """
    Convert a half-bar RN plan (len = 2*num_bars) into tonic-relative chord-tone sets.
    Each entry is a set of pitch classes in [0..11] relative to tonic.
    """
    tonic_pc = key_obj.tonic.pitchClass
    out: List[set[int]] = []

    for rn in rn_plan:
        rn = (rn or "N").strip()
        if rn == "N":
            out.append(set())
            continue

        # Normalise diminished symbol conventions
        rn_norm = rn.replace("°", "o").replace("ø", "o")

        try:
            rn_obj = m21.roman.RomanNumeral(rn_norm, key_obj)
            pcs_abs = set(rn_obj.pitchClasses)  # absolute pitch classes 0..11
            pcs_rel = {(pc - tonic_pc) % 12 for pc in pcs_abs}
            out.append(pcs_rel)
        except Exception:
            out.append(set())

    return out


def is_strong_beat(
    cur_unit: int,
    *,
    units_per_beat: int,
    beats_per_bar: int,
    strong_beats: Tuple[int, ...] = (0, 2),  # beats 1 and 3 (0-indexed)
) -> bool:
    beat_idx = (cur_unit // units_per_beat) % beats_per_bar
    return beat_idx in strong_beats


def halfbar_index_for_unit(
    cur_unit: int,
    *,
    units_per_beat: int,
) -> int:
    # half-bar = 2 beats = 2 * units_per_beat units
    halfbar_units = 2 * units_per_beat
    return cur_unit // halfbar_units


# ----------------------------
# Simple n-gram model
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
    counts_by_order: List[Dict[Context, Dict[Token, int]]]  # for orders 1..k

    def prob(self, tok: Token, ctx: Context) -> float:
        order = min(self.k, len(ctx) + 1)
        ctx = ctx[-(order - 1):] if order > 1 else ()

        # backoff if unseen context
        counts = self.counts_by_order[order - 1].get(ctx)
        if counts is None and order > 1:
            return self.prob(tok, ctx[1:])

        counts = counts or {}
        total = sum(counts.values())
        c = counts.get(tok, 0)
        V = len(self.vocab)
        return (c + self.alpha) / (total + self.alpha * V)

    def sample(self, ctx: Context, rng: random.Random) -> Token:
        ctx = ctx[-(self.k - 1):] if self.k > 1 else ()
        weights = [self.prob(t, ctx) for t in self.vocab]
        s = sum(weights)
        if s <= 0:
            return rng.choice(self.vocab)
        x = rng.random() * s
        acc = 0.0
        for t, w in zip(self.vocab, weights):
            acc += w
            if x <= acc:
                return t
        return self.vocab[-1]


def train_ngram(seqs: Iterable[Sequence[Token]], cfg: NGramConfig) -> NGramModel:
    vocab_set = set()
    seqs_list = []
    for s in seqs:
        # Remove BAR tokens if present
        s2 = [t for t in s if t != "BAR"]
        if not s2:
            continue
        seqs_list.append(s2)
        vocab_set.update(s2)

    vocab = sorted(vocab_set)
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
# Infill configuration
# ----------------------------

@dataclass
class InfillConfig:
    units_per_beat: int = 2
    # allowed quarterLength durations for generated tokens
    dur_set: Tuple[float, ...] = (2.0, 1.0, 0.5, 0.25)
    # prevent gaps being filled with too many rests
    max_consecutive_rests: int = 2
    seed: Optional[int] = None
    voice_leading_lambda: float = 1.0
    strong_beat_pick_closest: bool = False


def _allowed_tokens_for_remaining(
    vocab: Sequence[Token],
    remaining_units: int,
    cfg: InfillConfig,
) -> List[Token]:
    out: List[Token] = []
    for t in vocab:
        du = token_dur_units(t, cfg.units_per_beat)
        if du <= 0 or du > remaining_units:
            continue
        # Ensure duration is in dur_set (some datasets might contain other values)
        m = _NOTE_RE.match(t)
        if m:
            ql = float(m.group(2))
            if ql not in cfg.dur_set:
                continue
        m = _REST_RE.match(t)
        if m:
            ql = float(m.group(1))
            if ql not in cfg.dur_set:
                continue
        out.append(t)
    return out


def _sample_from_subset(model: NGramModel, ctx: Context, subset: Sequence[Token], rng: random.Random) -> Token:
    weights = [model.prob(t, ctx) for t in subset]
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


def _sample_with_voice_leading(
    model: NGramModel,
    ctx: Context,
    subset: Sequence[Token],
    rng: random.Random,
    prev_pc: Optional[int],
    lam: float,
) -> Token:
    """
    Sample from subset using n-gram probability multiplied by a voice-leading bonus:
      weight = P_ngram(tok|ctx) * exp(-lam * dist(prev_pc, tok_pc))
    Rests are not penalised.
    """
    import math

    weights: List[float] = []
    for t in subset:
        w = model.prob(t, ctx)
        if prev_pc is not None and is_note(t) and lam > 1e-9:
            pc = pc_of(t)
            if pc is not None:
                d = pc_circular_dist(prev_pc, pc)
                w *= math.exp(-lam * d)
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


# ----------------------------
# Timeline infilling
# ----------------------------

def infer_gaps(events: Sequence[Event], total_units: int) -> List[Tuple[int, int]]:
    """
    Compute uncovered gaps between event windows.
    Assumes events are motif windows already placed on the timeline.
    Returns list of (start_units, end_units) gaps.
    """
    evs = sorted(events, key=lambda e: e.start_units)
    gaps: List[Tuple[int, int]] = []
    cur = 0
    for e in evs:
        s = max(0, e.start_units)
        e_end = min(total_units, e.start_units + e.dur_units)
        if s > cur:
            gaps.append((cur, s))
        cur = max(cur, e_end)
    if cur < total_units:
        gaps.append((cur, total_units))
    return [(a, b) for (a, b) in gaps if b > a]


def fill_gap_tokens(
    *,
    model: NGramModel,
    context: List[Token],
    gap_units: int,
    gap_start_unit: int,
    cfg: InfillConfig,
    rng: random.Random,
    # --- harmony-aware args (optional) ---
    chord_pcs_by_halfbar: Optional[Sequence[set[int]]] = None,
    beats_per_bar: int = 4,
    strong_beats: Tuple[int, ...] = (0, 2),
) -> List[Token]:
    """
    Generate tokens to fill exactly gap_units.
    If chord_pcs_by_halfbar is provided, constrain NOTE tokens on strong beats
    to chord tones of the active half-bar chord.
    """
    out: List[Token] = []
    remaining = gap_units
    rest_run = 0
    cur_unit = gap_start_unit

    while remaining > 0:
        ctx = tuple(context[-(model.k - 1):]) if model.k > 1 else ()
        subset = _allowed_tokens_for_remaining(model.vocab, remaining_units=remaining, cfg=cfg)
        if not subset:
            # fallback: fill with smallest rest
            smallest = min(cfg.dur_set)
            du = max(1, int(round(smallest * cfg.units_per_beat)))
            tok = make_rest(smallest)
            out.append(tok)
            remaining -= du
            context.append(tok)
            rest_run += 1
            cur_unit += du
            continue

        # avoid too many consecutive rests
        if rest_run >= cfg.max_consecutive_rests:
            subset = [t for t in subset if not is_rest(t)] or subset

        # --- Harmony-aware filtering on strong beats ---
        if chord_pcs_by_halfbar is not None and is_strong_beat(
            cur_unit,
            units_per_beat=cfg.units_per_beat,
            beats_per_bar=beats_per_bar,
            strong_beats=strong_beats,
        ):
            hi = halfbar_index_for_unit(cur_unit, units_per_beat=cfg.units_per_beat)
            chord_pcs = chord_pcs_by_halfbar[hi] if 0 <= hi < len(chord_pcs_by_halfbar) else set()

            # If we have chord info, restrict NOTE tokens to chord tones.
            if chord_pcs:
                subset_notes = []
                subset_other = []
                for t in subset:
                    if is_note(t):
                        pc = pc_of(t)
                        if pc is not None and pc in chord_pcs:
                            subset_notes.append(t)
                    else:
                        subset_other.append(t)

                # Prefer chord-tone notes; if none, fall back to original subset
                if subset_notes:
                    subset = subset_notes + subset_other

        prev_pc = last_note_pc(context)

        # Prefer nearest chord tone on strong beats
        if (
            chord_pcs_by_halfbar is not None
            and prev_pc is not None
            and cfg.strong_beat_pick_closest
            and is_strong_beat(cur_unit, units_per_beat=cfg.units_per_beat, beats_per_bar=beats_per_bar, strong_beats=strong_beats)
        ):
            hi = halfbar_index_for_unit(cur_unit, units_per_beat=cfg.units_per_beat)
            chord_pcs = chord_pcs_by_halfbar[hi] if 0 <= hi < len(chord_pcs_by_halfbar) else set()
            if chord_pcs:
                # restrict to chord-tone notes that fit remaining
                chord_note_subset = [t for t in subset if is_note(t) and (pc_of(t) in chord_pcs)]
                if chord_note_subset:
                    # choose the closest pc deterministically, tie-break by n-gram prob
                    best = chord_note_subset[0]
                    best_d = pc_circular_dist(prev_pc, pc_of(best))
                    best_p = model.prob(best, ctx)
                    for t in chord_note_subset[1:]:
                        d = pc_circular_dist(prev_pc, pc_of(t))
                        p = model.prob(t, ctx)
                        if d < best_d or (d == best_d and p > best_p):
                            best, best_d, best_p = t, d, p
                    tok = best
                else:
                    tok = _sample_with_voice_leading(model, ctx, subset, rng, prev_pc, cfg.voice_leading_lambda)
            else:
                tok = _sample_with_voice_leading(model, ctx, subset, rng, prev_pc, cfg.voice_leading_lambda)
        else:
            # soft voice-leading weighting everywhere (default)
            tok = _sample_with_voice_leading(model, ctx, subset, rng, prev_pc, cfg.voice_leading_lambda)

        du = token_dur_units(tok, cfg.units_per_beat)
        if du <= 0 or du > remaining:
            continue

        out.append(tok)
        remaining -= du
        context.append(tok)
        rest_run = rest_run + 1 if is_rest(tok) else 0
        cur_unit += du

    return out


def infill_timeline_with_spans(
    *,
    events: Sequence[Event],
    total_units: int,
    motif_tokens_by_event: Dict[Tuple[int, int], List[Token]],
    model: NGramModel,
    cfg: InfillConfig,
    color_map: Dict[str, str],
    chord_pcs_by_halfbar: Optional[Sequence[set[int]]] = None,
    beats_per_bar: int = 4,
    strong_beats: Tuple[int, ...] = (0, 2),
) -> Tuple[List[Token], List[Tuple[int, int, str]]]:
    """
    Like infill_timeline(...), but also returns color spans for motif blocks
    so the final rendered Part can highlight motif instances.
    """
    rng = random.Random(cfg.seed)
    evs = sorted(events, key=lambda e: e.start_units)

    out: List[Token] = []
    context: List[Token] = []
    spans: List[Tuple[int, int, str]] = []
    cur = 0

    for e in evs:
        s = max(0, e.start_units)
        e_end = min(total_units, e.start_units + e.dur_units)

        # gap
        if s > cur:
            gap = s - cur
            out.extend(fill_gap_tokens(
                model=model,
                context=context,
                gap_units=gap,
                gap_start_unit=cur,
                cfg=cfg,
                rng=rng,
                chord_pcs_by_halfbar=chord_pcs_by_halfbar,
                beats_per_bar=beats_per_bar,
                strong_beats=strong_beats,
            ))
            cur = s

        # motif block
        key = (e.start_units, e.dur_units)
        block = motif_tokens_by_event.get(key, [])
        if not block:
            # fallback: fill with rests
            remaining = e_end - cur
            while remaining > 0:
                ql = min(cfg.dur_set)
                du = max(1, int(round(ql * cfg.units_per_beat)))
                if du > remaining:
                    ql = max(d for d in cfg.dur_set if int(round(d * cfg.units_per_beat)) <= remaining)
                    du = int(round(ql * cfg.units_per_beat))
                block.append(make_rest(ql))
                remaining -= du

        start_idx = len(out)
        out.extend(block)
        end_idx = len(out)

        # record span for colouring
        col = color_map.get(e.tok)
        if col is not None and end_idx > start_idx:
            spans.append((start_idx, end_idx, col))

        context.extend(block)
        cur = e_end

    # tail
    if cur < total_units:
        gap = total_units - cur
        out.extend(fill_gap_tokens(
            model=model,
            context=context,
            gap_units=gap,
            gap_start_unit=cur,
            cfg=cfg,
            rng=rng,
            chord_pcs_by_halfbar=chord_pcs_by_halfbar,
            beats_per_bar=beats_per_bar,
            strong_beats=strong_beats,
        ))

    # (optional) duration sanity is in infill_timeline; keep it simple here
    return out, spans