# src/motifgen/melody_ngram.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random
import re

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


def make_note(pc: int, ql: float) -> str:
    return f"N:{pc}:{ql}"


def make_rest(ql: float) -> str:
    return f"R:{ql}"


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
    dur_set: Tuple[float, ...] = (2.0, 1.0, 0.5)
    # prevent gaps being filled with too many rests
    max_consecutive_rests: int = 2
    seed: Optional[int] = None


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
    cfg: InfillConfig,
    rng: random.Random,
) -> List[Token]:
    """
    Generate tokens to fill exactly gap_units.
    Returns a list of N:/R: tokens.
    """
    out: List[Token] = []
    remaining = gap_units
    rest_run = 0

    while remaining > 0:
        ctx = tuple(context[-(model.k - 1):]) if model.k > 1 else ()
        subset = _allowed_tokens_for_remaining(model.vocab, remaining_units=remaining, cfg=cfg)
        if not subset:
            # fallback: fill with the smallest rest
            smallest = min(cfg.dur_set)
            du = max(1, int(round(smallest * cfg.units_per_beat)))
            out.append(make_rest(smallest))
            remaining -= du
            context.append(out[-1])
            rest_run += 1
            continue

        # avoid too many consecutive rests
        if rest_run >= cfg.max_consecutive_rests:
            subset = [t for t in subset if not is_rest(t)] or subset

        tok = _sample_from_subset(model, ctx, subset, rng)
        du = token_dur_units(tok, cfg.units_per_beat)
        if du <= 0 or du > remaining:
            continue

        out.append(tok)
        remaining -= du
        context.append(tok)
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
            out.extend(fill_gap_tokens(model=model, context=context, gap_units=gap, cfg=cfg, rng=rng))
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
        out.extend(fill_gap_tokens(model=model, context=context, gap_units=gap, cfg=cfg, rng=rng))

    # (optional) duration sanity is in infill_timeline; keep it simple here
    return out, spans