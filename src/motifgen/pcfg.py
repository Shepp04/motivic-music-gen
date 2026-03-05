
# src/motifgen/pcfg.py
# grammar rules + sampler -> plan tokens
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random


# ----------------------------
# Types / data models
# ----------------------------

Symbol = str  # nonterminal or terminal


@dataclass(frozen=True)
class GrammarRule:
    lhs: Symbol
    rhs: Tuple[Symbol, ...]
    prob: float  # should sum to 1.0 across rules with same lhs


@dataclass(frozen=True)
class Event:
    """
    A scheduled structure token to be realised into notes later.

    start_units/dur_units are in 'grid units' where:
      - beats_per_bar = 4
      - units_per_beat = 2 -> unit = 1/2 beat (eighth-note grid)
        e.g. 1 beat = 2 units, 1 bar = 8 units.
    """
    tok: Symbol          # e.g. "M0", "REP", "SEQ+2", "INV", "CAD"
    start_units: int     # inclusive
    dur_units: int       # positive


@dataclass(frozen=True)
class SectionSpec:
    """
    Specifies how many bars each form section should consume.
    Must sum to total bars.
    """
    v_bars: int
    f_bars: int
    e_bars: int


@dataclass(frozen=True)
class SamplerConfig:
    """
    Controls sampling behaviour (recursion guard, time grid, etc.).
    """
    beats_per_bar: int = 4
    units_per_beat: int = 2  # 2 = half-beat grid (eighth-note)
    max_recursion_depth: int = 8
    max_events_per_section: int = 64
    seed: Optional[int] = None


# ----------------------------
# PCFG definition
# ----------------------------

class PCFG:
    """
    Probabilistic context-free grammar over structure tokens.

    The grammar produces terminal tokens such as:
      - "M0", "REP", "SEQ+1", "SEQ+2", "INV", "RET", "CAD"

    Sampling is done in two stages:
      1) sample terminal token sequence from the grammar (tree expansion)
      2) schedule terminals onto a beat/grid timeline (Event list)
    """

    def __init__(self, rules: Sequence[GrammarRule], start_symbol: Symbol = "S"):
        self.start_symbol = start_symbol
        self.rules_by_lhs: Dict[Symbol, List[GrammarRule]] = {}
        for r in rules:
            self.rules_by_lhs.setdefault(r.lhs, []).append(r)
        self._validate_probabilities()

    def _validate_probabilities(self) -> None:
        """
        Basic validation: per-LHS probabilities sum approximately to 1.
        """
        for lhs, rs in self.rules_by_lhs.items():
            s = sum(r.prob for r in rs)
            if not (0.999 <= s <= 1.001):
                raise ValueError(f"PCFG rules for {lhs} probs sum to {s}, expected 1.0")

    # -------- Sampling: grammar expansion --------

    def sample_terminals(
        self,
        *,
        cfg: SamplerConfig,
        rng: Optional[random.Random] = None,
    ) -> List[Symbol]:
        """
        Sample a sequence of terminal symbols by expanding from the start symbol.

        This does NOT assign times/durations. Use `schedule_terminals()` afterwards.
        """
        if rng is None:
            rng = random.Random(cfg.seed)

        # Expand start symbol into a sentential form and stop when only terminals remain.
        # Terminals are symbols with no outgoing rules in rules_by_lhs.
        sentential: List[Symbol] = [self.start_symbol]
        depth = 0

        while True:
            # Find first nonterminal (symbol that has rules)
            idx = next((i for i, s in enumerate(sentential) if s in self.rules_by_lhs), None)
            if idx is None:
                break  # all terminals

            if depth >= cfg.max_recursion_depth:
                # Hard stop: replace remaining nonterminals with a safe terminal (e.g., M0)
                sentential = [s if s not in self.rules_by_lhs else "M0" for s in sentential]
                break

            nt = sentential[idx]
            rule = self._sample_rule(nt, rng)
            # Replace nt with rhs
            sentential = sentential[:idx] + list(rule.rhs) + sentential[idx + 1 :]
            depth += 1

        return sentential

    def _sample_rule(self, lhs: Symbol, rng: random.Random) -> GrammarRule:
        rs = self.rules_by_lhs[lhs]
        # Weighted choice
        x = rng.random()
        acc = 0.0
        for r in rs:
            acc += r.prob
            if x <= acc:
                return r
        return rs[-1]

    # -------- Scheduling: terminals -> timed Events --------

    def schedule_terminals(
        self,
        terminals: Sequence[Symbol],
        *,
        section: SectionSpec,
        cfg: SamplerConfig,
        motif_dur_units: Sequence[int] = (4,),  # default: 2 beats = 4 half-beat units
        motif_dur_probs: Optional[Sequence[float]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[Event]:
        """
        Convert terminal token sequence into a time-ordered Event list.

        The scheduler assigns each motif-like token a duration sampled from motif_dur_units,
        then places them sequentially into V, F, E sections with simple guards.

        NOTE: This is intentionally simple and deterministic-friendly. You can later
        add stronger beat-biasing and cadence enforcement.
        """
        if rng is None:
            rng = random.Random(cfg.seed)

        units_per_bar = cfg.beats_per_bar * cfg.units_per_beat
        v_end = section.v_bars * units_per_bar
        f_end = v_end + section.f_bars * units_per_bar
        e_end = f_end + section.e_bars * units_per_bar

        # If no probs supplied, uniform
        if motif_dur_probs is None:
            motif_dur_probs = [1.0 / len(motif_dur_units)] * len(motif_dur_units)

        def sample_dur(tok: Symbol) -> int:
            # Cadence tokens can be slightly longer if desired; keep simple for MVP.
            if tok == "CAD":
                return max(motif_dur_units)
            # Weighted sample
            x = rng.random()
            acc = 0.0
            for d, p in zip(motif_dur_units, motif_dur_probs):
                acc += p
                if x <= acc:
                    return int(d)
            return int(motif_dur_units[-1])

        events: List[Event] = []
        t = 0

        def emit_in_window(start_t: int, end_t: int, toks: Sequence[Symbol]) -> None:
            nonlocal events
            nonlocal t
            t = start_t
            for tok in toks:
                if len(events) >= cfg.max_events_per_section:
                    break
                d = sample_dur(tok)
                if t + d > end_t:
                    break
                events.append(Event(tok=tok, start_units=t, dur_units=d))
                t += d

        # Naive mapping: consume terminals left-to-right:
        # first chunk into V, then F, then E. (You can replace with explicit V/F/E tokens later.)
        # MVP strategy: split terminals approximately by section lengths.
        total_units = e_end
        v_units = v_end
        f_units = f_end - v_end
        e_units = e_end - f_end

        n = len(terminals)
        if n == 0:
            return []

        # proportional split by section duration
        v_n = max(1, round(n * (v_units / total_units)))
        f_n = max(1, round(n * (f_units / total_units)))
        e_n = max(1, n - v_n - f_n)

        v_toks = list(terminals[:v_n])
        f_toks = list(terminals[v_n : v_n + f_n])
        e_toks = list(terminals[v_n + f_n :])

        # Ensure exactly one CAD, placed in E (at the end)
        v_toks = [t for t in v_toks if t != "CAD"]
        f_toks = [t for t in f_toks if t != "CAD"]
        e_toks = [t for t in e_toks if t != "CAD"]

        # Force a single cadence token at the end of E
        e_toks.append("CAD")

        emit_in_window(0, v_end, v_toks)
        emit_in_window(v_end, f_end, f_toks)
        emit_in_window(f_end, e_end, e_toks)

        return events

    # -------- Convenience: one-shot plan sampler --------

    def sample_plan(
        self,
        *,
        num_bars: int,
        section: Optional[SectionSpec] = None,
        cfg: Optional[SamplerConfig] = None,
        rng: Optional[random.Random] = None,
        motif_dur_units: Sequence[int] = (4,),  # 2 beats on half-beat grid
        motif_dur_probs: Optional[Sequence[float]] = None,
    ) -> Sequence[Event]:
        """
        One-shot plan sampler:
          - samples terminals from the grammar
          - schedules them into (V,F,E) sections across num_bars

        section defaults to a simple 2/5/1 split for 8 bars, scaled if needed.
        """
        if cfg is None:
            cfg = SamplerConfig()
        if rng is None:
            rng = random.Random(cfg.seed)

        if section is None:
            # Default split: 2 bars V, (num_bars-3) bars F, 1 bar E (min 1)
            v = max(1, min(2, num_bars - 2))
            e = 1
            f = max(1, num_bars - v - e)
            section = SectionSpec(v_bars=v, f_bars=f, e_bars=e)

        if section.v_bars + section.f_bars + section.e_bars != num_bars:
            raise ValueError("SectionSpec bars must sum to num_bars")

        terminals = self.sample_terminals(cfg=cfg, rng=rng)
        return self.schedule_terminals(
            terminals,
            section=section,
            cfg=cfg,
            motif_dur_units=motif_dur_units,
            motif_dur_probs=motif_dur_probs,
            rng=rng,
        )


# ----------------------------
# Minimal grammar factory (MVP)
# ----------------------------

def make_minimal_mvp_grammar() -> PCFG:
    """
    Returns a small, stable grammar. Terminals are:
      M0, REP, SEQ+1, SEQ+2, INV, CAD

    Nonterminals:
      S, V, F, E, SPIN
    """
    rules = [
        GrammarRule("S", ("V", "F", "E"), 1.0),

        GrammarRule("V", ("M0", "REP"), 0.6),
        GrammarRule("V", ("M0",), 0.4),

        GrammarRule("F", ("SPIN",), 1.0),

        GrammarRule("SPIN", ("SPIN", "SPIN"), 0.25),
        GrammarRule("SPIN", ("SEQ+1",), 0.15),
        GrammarRule("SPIN", ("SEQ+2",), 0.15),
        GrammarRule("SPIN", ("INV",), 0.15),
        GrammarRule("SPIN", ("RET",), 0.15),
        GrammarRule("SPIN", ("M0",), 0.15),

        GrammarRule("E", ("CAD",), 1.0),
    ]
    return PCFG(rules=rules, start_symbol="S")

if __name__ == "__main__":
    # Quick test
    grammar = make_minimal_mvp_grammar()
    cfg = SamplerConfig(seed=random.randint(0, 1000000))#seed=42)
    plan = grammar.sample_plan(num_bars=8, cfg=cfg)
    for e in plan:
        print(e)