# src/motifgen/pcfg.py
# grammar rules + sampler -> plan tokens
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random

Symbol = str


@dataclass(frozen=True)
class GrammarRule:
    lhs: Symbol
    rhs: Tuple[Symbol, ...]
    prob: float  # sum to 1.0 per LHS


@dataclass(frozen=True)
class Event:
    """
    Scheduled structure token.

    start_units/dur_units are in grid units:
      - units_per_beat = 4 => 1 unit = semiquaver
      - beats_per_bar = 4 => units_per_bar = 16
    """
    tok: Symbol
    start_units: int
    dur_units: int


@dataclass(frozen=True)
class SectionSpec:
    v_bars: int
    f_bars: int
    e_bars: int


@dataclass(frozen=True)
class SamplerConfig:
    beats_per_bar: int = 4
    units_per_beat: int = 4
    max_recursion_depth: int = 10
    max_events_total: int = 256
    seed: Optional[int] = None

    # density: 0 -> sparse (more infill), 1 -> dense (more motif tokens)
    density: float = 0.5

    min_gap_units: int = 0
    max_gap_units: Optional[int] = None


class PCFG:
    def __init__(self, rules: Sequence[GrammarRule], start_symbol: Symbol = "S"):
        self.start_symbol = start_symbol
        self.rules_by_lhs: Dict[Symbol, List[GrammarRule]] = {}
        for r in rules:
            self.rules_by_lhs.setdefault(r.lhs, []).append(r)
        self._validate()

    def _validate(self) -> None:
        for lhs, rs in self.rules_by_lhs.items():
            s = sum(r.prob for r in rs)
            if not (0.999 <= s <= 1.001):
                raise ValueError(f"PCFG rules for {lhs} probs sum to {s}, expected 1.0")

    def _sample_rule(self, lhs: Symbol, rng: random.Random) -> GrammarRule:
        rs = self.rules_by_lhs[lhs]
        x = rng.random()
        acc = 0.0
        for r in rs:
            acc += r.prob
            if x <= acc:
                return r
        return rs[-1]

    # ----------------------------
    # Stage 1: expand -> terminals
    # ----------------------------

    def sample_terminals(self, *, cfg: SamplerConfig, rng: Optional[random.Random] = None) -> List[Symbol]:
        rng = rng or random.Random(cfg.seed)
        sentential: List[Symbol] = [self.start_symbol]

        depth = 0
        while True:
            idx = next((i for i, s in enumerate(sentential) if s in self.rules_by_lhs), None)
            if idx is None:
                break

            if depth >= cfg.max_recursion_depth:
                # replace remaining nonterminals with safe terminal
                sentential = [s if s not in self.rules_by_lhs else "M0" for s in sentential]
                break

            nt = sentential[idx]
            rule = self._sample_rule(nt, rng)
            sentential = sentential[:idx] + list(rule.rhs) + sentential[idx + 1 :]
            depth += 1

        return sentential

    # ----------------------------
    # Stage 2: schedule -> events
    # ----------------------------

    def schedule_terminals(
        self,
        terminals: Sequence[Symbol],
        *,
        num_bars: int,
        section: SectionSpec,
        cfg: SamplerConfig,
        motif_dur_units: Sequence[int],
        motif_dur_probs: Optional[Sequence[float]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[Event]:
        rng = rng or random.Random(cfg.seed)
        density = max(0.0, min(1.0, cfg.density))

        units_per_bar = cfg.beats_per_bar * cfg.units_per_beat
        total_units = num_bars * units_per_bar

        v_end = section.v_bars * units_per_bar
        f_end = v_end + section.f_bars * units_per_bar
        e_end = total_units  # section.e_bars * units_per_bar but we use total_units directly

        # ---- cadence: force final bar, beat-aligned ----
        cad_dur = units_per_bar
        cad_start = total_units - cad_dur

        # If no terminals, still return a cadence
        if not terminals:
            return [Event("CAD", cad_start, cad_dur)]

        # ---- validate duration distribution ----
        if motif_dur_probs is None:
            motif_dur_probs = [1.0 / len(motif_dur_units)] * len(motif_dur_units)
        if len(motif_dur_probs) != len(motif_dur_units):
            raise ValueError("motif_dur_probs must be same length as motif_dur_units")
        # normalize probs (safe)
        ps = list(motif_dur_probs)
        s = sum(ps)
        if s <= 0:
            ps = [1.0 / len(ps)] * len(ps)
        else:
            ps = [p / s for p in ps]

        def sample_dur(tok: Symbol) -> int:
            if tok == "CAD":
                return cad_dur
            x = rng.random()
            acc = 0.0
            for d, p in zip(motif_dur_units, ps):
                acc += p
                if x <= acc:
                    return int(d)
            return int(motif_dur_units[-1])

        def snap_to_beat(u: int) -> int:
            # floor to beat boundary
            return (u // cfg.units_per_beat) * cfg.units_per_beat

        def beat_gap(max_units: int) -> int:
            # sample a gap in whole beats (prevents off-beat motif starts)
            if max_units <= 0:
                return 0
            max_beats = max_units // cfg.units_per_beat
            if max_beats <= 0:
                return 0

            # density bias: high density -> small gaps
            p_small = 0.65 + 0.30 * density
            if rng.random() < p_small:
                beats = 0
            else:
                beats = rng.randint(0, max_beats)
            gap = beats * cfg.units_per_beat

            # apply min_gap_units (also beat-aligned)
            min_gap = snap_to_beat(max(0, cfg.min_gap_units))
            gap = max(min_gap, gap)
            return gap

        # max gap policy
        if cfg.max_gap_units is None:
            # low density allows bigger gaps, high density clamps
            cfg_max_gap = int(round((1.5 - 1.2 * density) * units_per_bar))  # ~12..2 (for upb=2); scales OK for upb=4
            cfg_max_gap = max(cfg.units_per_beat, cfg_max_gap)
        else:
            cfg_max_gap = max(cfg.units_per_beat, int(cfg.max_gap_units))

        # ---- split terminals into V/F/E chunks (proportional) ----
        # remove any CAD in terminals; we force it ourselves
        toks = [t for t in terminals if t != "CAD"]

        # Expand SEQUPk/SEQDNk into multiple SEQ steps (bounded)
        def expand_seq_runs(toks_in: List[Symbol], max_seq_events: int) -> List[Symbol]:
            out: List[Symbol] = []
            used = 0
            fallback = ["M0", "INV", "RET", "REP"]

            for t in toks_in:
                if t.startswith("SEQUP"):
                    k = int(t.replace("SEQUP", ""))
                    if used + k <= max_seq_events:
                        out.extend([f"SEQ+{i}" for i in range(1, k + 1)])
                        used += k
                    else:
                        out.append(rng.choice(fallback))
                elif t.startswith("SEQDN"):
                    k = int(t.replace("SEQDN", ""))
                    if used + k <= max_seq_events:
                        out.extend([f"SEQ-{i}" for i in range(1, k + 1)])
                        used += k
                    else:
                        out.append(rng.choice(fallback))
                else:
                    out.append(t)
            return out

        # allocate a sequence quota mostly to fortspinnung
        max_seq_events = max(1, int(round((0.25 + 0.45 * density) * (section.f_bars * 2))))
        toks = expand_seq_runs(toks, max_seq_events=max_seq_events)

        # proportional split by section duration
        v_units = v_end
        f_units = f_end - v_end
        e_units = cad_start - f_end  # pre-cadence window only (epilog is CAD)

        total_pre_cad = max(1, v_units + f_units + max(0, e_units))
        n = len(toks)
        v_n = max(1, round(n * (v_units / total_pre_cad)))
        f_n = max(1, round(n * (f_units / total_pre_cad)))
        v_n = min(v_n, n)
        f_n = min(f_n, n - v_n)
        e_n = max(0, n - v_n - f_n)

        v_toks = toks[:v_n]
        f_toks = toks[v_n : v_n + f_n]
        e_toks = toks[v_n + f_n : v_n + f_n + e_n]

        # ---- scheduling policy ----
        # Coverage fraction: fraction of each section window to fill with motif events.
        coverage = 0.25 + 0.55 * density  # 0.25..0.80

        events: List[Event] = []

        def emit_with_gaps(start: int, end: int, toks_in: List[Symbol]) -> None:
            """Sequential placement with beat-aligned gaps."""
            if start >= end or not toks_in:
                return
            t = snap_to_beat(start)
            end = snap_to_beat(end)

            # motif time budget for this window
            window = max(0, end - t)
            budget = int(round(window * coverage))

            used = 0
            i = 0
            while i < len(toks_in) and t < end and used < budget and len(events) < cfg.max_events_total:
                tok = toks_in[i]
                d = sample_dur(tok)
                d = snap_to_beat(max(cfg.units_per_beat, d))  # ensure whole-beat durations

                if t + d > end:
                    break

                events.append(Event(tok=tok, start_units=t, dur_units=d))
                used += d
                t += d
                i += 1

                if t >= end or used >= budget:
                    break

                gap_cap = min(cfg_max_gap, end - t)
                t += beat_gap(gap_cap)

        def emit_spread(start: int, end: int, toks_in: List[Symbol], events_per_bar: float) -> None:
            """Spread motif events across the whole window (fortspinnung coverage)."""
            if start >= end or not toks_in:
                return
            start = snap_to_beat(start)
            end = snap_to_beat(end)
            window = end - start
            if window <= 0:
                return

            bars = window / float(units_per_bar)
            n_events = max(1, int(round(bars * events_per_bar)))
            step = window / float(n_events)

            for k in range(n_events):
                if len(events) >= cfg.max_events_total:
                    break

                tok = toks_in[k % len(toks_in)]
                d = sample_dur(tok)
                d = snap_to_beat(max(cfg.units_per_beat, d))

                # anchor + small jitter, then snap
                anchor = start + int(round(k * step))
                jitter = int(round((rng.random() - 0.5) * min(step * 0.35, cfg_max_gap)))
                t = snap_to_beat(max(start, min(end - d, anchor + jitter)))

                if t + d > end:
                    continue
                events.append(Event(tok=tok, start_units=t, dur_units=d))

        # V: sequential with gaps
        emit_with_gaps(0, v_end, v_toks)

        # F: spread across whole fortspinnung (prevents “1–2 bars of motif then silence”)
        f_events_per_bar = 0.75 + 1.25 * density  # ~0.75..2.0
        emit_spread(v_end, f_end, f_toks, events_per_bar=f_events_per_bar)

        # Pre-cadence epilog material
        if cad_start > f_end:
            emit_with_gaps(f_end, cad_start, e_toks)

        # Force cadence last, aligned
        events.append(Event("CAD", cad_start, cad_dur))

        # Final: ensure sorted and strictly beat-aligned
        events.sort(key=lambda e: e.start_units)
        events = [Event(e.tok, snap_to_beat(e.start_units), snap_to_beat(e.dur_units)) for e in events]

        # Ensure cadence is last and ends exactly at total_units
        if events[-1].tok != "CAD":
            events.append(Event("CAD", cad_start, cad_dur))
        else:
            last = events[-1]
            events[-1] = Event("CAD", cad_start, cad_dur)

        return events

    def sample_plan(
        self,
        *,
        num_bars: int,
        cfg: SamplerConfig,
        section: Optional[SectionSpec] = None,
        motif_dur_units: Sequence[int] = (8,),
        motif_dur_probs: Optional[Sequence[float]] = None,
    ) -> List[Event]:
        rng = random.Random(cfg.seed)

        if section is None:
            # default: 2 bars V, 1 bar E, rest F
            v = max(1, min(2, num_bars - 2))
            e = 1
            f = max(1, num_bars - v - e)
            section = SectionSpec(v_bars=v, f_bars=f, e_bars=e)

        if section.v_bars + section.f_bars + section.e_bars != num_bars:
            raise ValueError("SectionSpec bars must sum to num_bars")

        terminals = self.sample_terminals(cfg=cfg, rng=rng)
        return self.schedule_terminals(
            terminals,
            num_bars=num_bars,
            section=section,
            cfg=cfg,
            motif_dur_units=motif_dur_units,
            motif_dur_probs=motif_dur_probs,
            rng=rng,
        )


# ----------------------------
# Grammar
# ----------------------------

def make_grammar(density: float = 0.5) -> PCFG:
    """
    Vordersatz -> Fortspinnung -> Epilog (CAD), with sequence runs.
    Density increases F recursion (more CELLs).
    """
    density = max(0.0, min(1.0, density))
    p_more = 0.25 + 0.55 * density
    p_one = 1.0 - p_more

    rules = [
        GrammarRule("S", ("V", "F", "E"), 1.0),

        GrammarRule("V", ("M0", "REP"), 0.55),
        GrammarRule("V", ("M0",), 0.45),

        GrammarRule("F", ("CELL", "F"), p_more),
        GrammarRule("F", ("CELL",), p_one),

        GrammarRule("CELL", ("M0",), 0.14),
        GrammarRule("CELL", ("INV",), 0.12),
        GrammarRule("CELL", ("RET",), 0.10),
        GrammarRule("CELL", ("REP",), 0.08),
        GrammarRule("CELL", ("SEQUP2",), 0.16),
        GrammarRule("CELL", ("SEQUP3",), 0.12),
        GrammarRule("CELL", ("SEQDN2",), 0.16),
        GrammarRule("CELL", ("SEQDN3",), 0.12),

        GrammarRule("E", ("CAD",), 1.0),
    ]
    return PCFG(rules, start_symbol="S")


if __name__ == "__main__":
    g = make_grammar(density=0.6)
    cfg = SamplerConfig(seed=65, units_per_beat=4, density=0.6, min_gap_units=0)
    plan = g.sample_plan(num_bars=8, cfg=cfg, motif_dur_units=(8,))
    for e in plan:
        print(e)