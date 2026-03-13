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

Mode = str  # "major" | "minor"
MAJOR_PCS = {0, 2, 4, 5, 7, 9, 11}
HARM_MINOR_PCS = {0, 2, 3, 5, 7, 8, 10, 11}  # raised 7 allowed


# ----------------------------
# Token utilities
# ----------------------------

def is_note(tok: str) -> bool:
    return _NOTE_RE.match(tok) is not None


def is_rest(tok: str) -> bool:
    return _REST_RE.match(tok) is not None


def pc_of(tok: str) -> Optional[int]:
    m = _NOTE_RE.match(tok)
    return int(m.group(1)) if m else None


def ql_of(tok: str) -> Optional[float]:
    m = _NOTE_RE.match(tok)
    if m:
        return float(m.group(2))
    m = _REST_RE.match(tok)
    if m:
        return float(m.group(1))
    return None


def token_dur_units(tok: str, units_per_beat: int) -> int:
    ql = ql_of(tok)
    if ql is None:
        return 0
    return max(1, int(round(ql * units_per_beat)))


def last_note_pc(tokens: Sequence[str]) -> Optional[int]:
    for t in reversed(tokens):
        if is_note(t):
            return pc_of(t)
    return None


def pc_dist(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


def allowed_note_pcs(mode: Mode) -> set[int]:
    return set(MAJOR_PCS) if mode == "major" else set(HARM_MINOR_PCS)


def halfbar_index_for_unit(cur_unit: int, *, units_per_beat: int) -> int:
    return cur_unit // (2 * units_per_beat)


def is_strong_beat(cur_unit: int, *, units_per_beat: int, beats_per_bar: int, strong_beats: Tuple[int, ...]) -> bool:
    beat = (cur_unit // units_per_beat) % beats_per_bar
    return beat in strong_beats


def _rn_root_is_V(rn: str) -> bool:
    if not rn or rn == "N":
        return False
    s = rn.strip().replace("°", "o").replace("ø", "o").split("/")[0]
    m = _RN_ROOT_RE.match(s)
    return bool(m and m.group(1) == "V")


# ----------------------------
# Harmony utilities
# ----------------------------


def chord_pc_sets_from_rn_plan(rn_plan: Sequence[str], *, key_obj: m21.key.Key) -> List[set[int]]:
    tonic_pc = key_obj.tonic.pitchClass
    out: List[set[int]] = []
    for rn in rn_plan:
        rn = (rn or "N").strip()
        if rn == "N":
            out.append(set())
            continue
        try:
            rn_obj = m21.roman.RomanNumeral(rn.replace("°", "o").replace("ø", "o"), key_obj)
            out.append({(pc - tonic_pc) % 12 for pc in rn_obj.pitchClasses})
        except Exception:
            out.append(set())
    return out


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

    def sample_weighted(
        self,
        ctx: Context,
        rng: random.Random,
        subset: Sequence[Token],
        prev_pc: Optional[int],
        lam: float,
    ) -> Token:
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
# Infill configuration
# ----------------------------

@dataclass
class InfillConfig:
    units_per_beat: int = 4
    dur_set: Tuple[float, ...] = (2.0, 1.0, 0.5, 0.25)

    allow_rests: bool = True
    min_rest_ql: float = 1.0
    max_consecutive_rests: int = 2

    seed: Optional[int] = None
    voice_leading_lambda: float = 0.9

    strong_beats: Tuple[int, ...] = (0, 2)  # beats 1 and 3
    beats_per_bar: int = 4
    strong_beat_pick_closest: bool = False


# ----------------------------
# Core filtering + block sizing
# ----------------------------

def _fits(tok: Token, remaining_units: int, cfg: InfillConfig) -> bool:
    du = token_dur_units(tok, cfg.units_per_beat)
    if du <= 0 or du > remaining_units:
        return False
    ql = ql_of(tok)
    return (ql is not None) and (ql in cfg.dur_set)


def _block_units(block: Sequence[Token], units_per_beat: int) -> int:
    return sum(token_dur_units(t, units_per_beat) for t in block)


def _trim_or_pad_block(block: List[Token], target_units: int, cfg: InfillConfig) -> List[Token]:
    """
    HARD GUARD: ensure block duration == target_units (prevents total_units drift).
    - Too long: pop tokens from end
    - Too short: pad with rests (or tonic notes if rests disabled)
    """
    out = list(block)
    cur = _block_units(out, cfg.units_per_beat)

    while out and cur > target_units:
        cur -= token_dur_units(out.pop(), cfg.units_per_beat)

    while cur < target_units:
        rem = target_units - cur
        # choose largest ql that fits rem (so we don't spam tiny tokens)
        chosen = None
        for ql in sorted(cfg.dur_set, reverse=True):
            if int(round(ql * cfg.units_per_beat)) <= rem:
                chosen = ql
                break
        if chosen is None:
            chosen = min(cfg.dur_set)

        tok = f"R:{chosen}" if cfg.allow_rests else f"N:0:{chosen}"
        out.append(tok)
        cur += token_dur_units(tok, cfg.units_per_beat)

    return out


# ----------------------------
# Gap filler
# ----------------------------

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
    allowed_pcs = allowed_note_pcs(mode)
    key_obj = key_obj or m21.key.Key("C", "minor" if mode == "minor" else "major")

    out: List[Token] = []
    remaining = gap_units
    rest_run = 0
    cur_unit = gap_start_unit

    while remaining > 0:
        ctx = tuple(context[-(model.k - 1):]) if model.k > 1 else ()
        prev_pc = last_note_pc(context)

        # 1) duration filter
        cand = [t for t in model.vocab if _fits(t, remaining, cfg)]
        if not cand:
            # smallest safe pad
            ql = min(cfg.dur_set)
            tok = f"R:{ql}" if cfg.allow_rests else f"N:0:{ql}"
            du = token_dur_units(tok, cfg.units_per_beat)
            out.append(tok)
            context.append(tok)
            remaining -= du
            cur_unit += du
            rest_run = rest_run + 1 if is_rest(tok) else 0
            continue

        # 2) rest policy
        if not cfg.allow_rests:
            nonrests = [t for t in cand if not is_rest(t)]
            cand = nonrests or cand
        else:
            # forbid short rests after the first token of a gap (avoid syncopated micro-rests)
            if cur_unit != gap_start_unit:
                cand2 = []
                for t in cand:
                    if is_rest(t) and (ql_of(t) or 0.0) < cfg.min_rest_ql:
                        continue
                    cand2.append(t)
                cand = cand2 or cand

        if rest_run >= cfg.max_consecutive_rests:
            nonrests = [t for t in cand if not is_rest(t)]
            cand = nonrests or cand

        # 3) mode constraint on ALL notes
        cand2 = []
        for t in cand:
            if is_note(t):
                pc = pc_of(t)
                if pc is not None and pc in allowed_pcs:
                    cand2.append(t)
            else:
                cand2.append(t)
        cand = cand2 or cand

        # 4) minor-over-V raised scale degrees filter (forbid b6/b7 pcs 8/10 under V)
        if rn_plan is not None and (key_obj.mode or "").lower() == "minor":
            hi = halfbar_index_for_unit(cur_unit, units_per_beat=cfg.units_per_beat)
            rn = rn_plan[hi] if 0 <= hi < len(rn_plan) else "N"
            if _rn_root_is_V(rn):
                cand2 = [t for t in cand if not (is_note(t) and (pc_of(t) in {8, 10}))]
                cand = cand2 or cand

        # 5) strong-beat chord tone constraint (optional)
        chord_pcs = set()
        strong = False
        if chord_pcs_by_halfbar is not None and is_strong_beat(
            cur_unit,
            units_per_beat=cfg.units_per_beat,
            beats_per_bar=cfg.beats_per_bar,
            strong_beats=cfg.strong_beats,
        ):
            strong = True
            hi = halfbar_index_for_unit(cur_unit, units_per_beat=cfg.units_per_beat)
            chord_pcs = chord_pcs_by_halfbar[hi] if 0 <= hi < len(chord_pcs_by_halfbar) else set()

            if chord_pcs:
                notes = [t for t in cand if is_note(t) and (pc_of(t) in chord_pcs)]
                others = [t for t in cand if not is_note(t)]
                cand = (notes + others) if notes else cand

        # 6) choose token
        if cfg.strong_beat_pick_closest and strong and chord_pcs and prev_pc is not None:
            chord_notes = [t for t in cand if is_note(t) and (pc_of(t) in chord_pcs)]
            if chord_notes:
                tok = min(chord_notes, key=lambda t: (pc_dist(prev_pc, pc_of(t)), -model.prob(t, ctx)))
            else:
                tok = model.sample_weighted(ctx, rng, cand, prev_pc, cfg.voice_leading_lambda)
        else:
            tok = model.sample_weighted(ctx, rng, cand, prev_pc, cfg.voice_leading_lambda)

        du = token_dur_units(tok, cfg.units_per_beat)
        if du <= 0 or du > remaining:
            continue

        out.append(tok)
        context.append(tok)
        remaining -= du
        cur_unit += du
        rest_run = rest_run + 1 if is_rest(tok) else 0

    return out


# ----------------------------
# Timeline assembly (with spans)
# ----------------------------

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
    ctx: List[Token] = []
    spans: List[Tuple[int, int, str]] = []
    cur = 0

    for e in evs:
        s = max(0, e.start_units)
        e_end = min(total_units, e.start_units + e.dur_units)

        if s < cur:
            # overlap with already-filled timeline; skip this event (MVP policy)
            continue

        # gap before event
        if s > cur:
            out.extend(
                fill_gap_tokens(
                    model=model,
                    context=ctx,
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

        # motif block for this event
        block = list(motif_tokens_by_event.get((e.start_units, e.dur_units), []))

        # HARD GUARD: ensure block sums to exactly e.dur_units
        block = _trim_or_pad_block(block, e.dur_units, cfg)

        a = len(out)
        out.extend(block)
        b = len(out)

        col = color_map.get(e.tok)
        if col and b > a:
            spans.append((a, b, col))

        ctx.extend(block)
        cur = e_end

    # tail
    if cur < total_units:
        out.extend(
            fill_gap_tokens(
                model=model,
                context=ctx,
                gap_units=total_units - cur,
                gap_start_unit=cur,
                cfg=cfg,
                rng=rng,
                mode=mode,
                chord_pcs_by_halfbar=chord_pcs_by_halfbar,
                rn_plan=rn_plan,
                key_obj=key_obj,
            )
        )

    # final sanity: prevents "expected 128 got 132" from ever returning silently
    total = _block_units(out, cfg.units_per_beat)
    if total != total_units:
        raise ValueError(f"infill_timeline_with_spans duration mismatch: got {total}, expected {total_units}")

    return out, spans