# src/motifgen/pcfg.py
# grammar rules + sampler -> plan tokens

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random


Symbol = str  # nonterminal or terminal


@dataclass(frozen=True)
class GrammarRule:
    lhs: Symbol
    rhs: Tuple[Symbol, ...]
    prob: float  # should sum to 1.0 across rules with same lhs


@dataclass(frozen=True)
class Event:
    """
    Scheduled structure token to be realised later.

    Grid:
      - beats_per_bar = 4
      - units_per_beat = 2 -> unit = 1/2 beat (eighth-note grid)
      - 1 bar = beats_per_bar * units_per_beat units (default 8)
    """
    tok: Symbol
    start_units: int
    dur_units: int


@dataclass(frozen=True)
class SectionSpec:
    """
    Bars allocated to each form section. Must sum to total bars.
    """
    v_bars: int
    f_bars: int
    e_bars: int


@dataclass(frozen=True)
class SamplerConfig:
    """
    Sampling behaviour (recursion guard, time grid, density controls).
    """
    beats_per_bar: int = 4
    units_per_beat: int = 2
    max_recursion_depth: int = 10
    max_events_per_section: int = 64
    seed: Optional[int] = None

    # Density controls (0=sparse -> many/sizable infill gaps, 1=dense -> few/small gaps)
    density: float = 0.5

    # Optional overrides. If None, derived from density.
    max_gap_units: Optional[int] = None
    min_gap_units: int = 0  # set to 1 if you always want at least half-beat gaps


class PCFG:
    """
    Probabilistic CFG over structure tokens, sampled in two stages:
      1) expand grammar -> terminal token list
      2) schedule tokens -> timed Events with density-controlled gaps
    """

    def __init__(self, rules: Sequence[GrammarRule], start_symbol: Symbol = "S"):
        self.start_symbol = start_symbol
        self.rules_by_lhs: Dict[Symbol, List[GrammarRule]] = {}
        for r in rules:
            self.rules_by_lhs.setdefault(r.lhs, []).append(r)
        self._validate_probabilities()

    def _validate_probabilities(self) -> None:
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
    # Stage 1: grammar expansion
    # ----------------------------

    def sample_terminals(self, *, cfg: SamplerConfig, rng: Optional[random.Random] = None) -> List[Symbol]:
        if rng is None:
            rng = random.Random(cfg.seed)

        sentential: List[Symbol] = [self.start_symbol]
        depth = 0

        while True:
            idx = next((i for i, s in enumerate(sentential) if s in self.rules_by_lhs), None)
            if idx is None:
                break  # all terminals

            if depth >= cfg.max_recursion_depth:
                # Hard stop: replace remaining nonterminals with safe terminals
                sentential = [s if s not in self.rules_by_lhs else "M0" for s in sentential]
                break

            nt = sentential[idx]
            rule = self._sample_rule(nt, rng)
            sentential = sentential[:idx] + list(rule.rhs) + sentential[idx + 1 :]
            depth += 1

        return sentential

    # ----------------------------
    # Stage 2: scheduling
    # ----------------------------

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
        Density-aware scheduling:
          - events are placed sequentially within each section window
          - bounded gaps are inserted between events
          - section motif coverage is controlled by cfg.density
          - a single CAD is forced at the very end
        """
        if rng is None:
            rng = random.Random(cfg.seed)

        density = max(0.0, min(1.0, cfg.density))

        units_per_bar = cfg.beats_per_bar * cfg.units_per_beat
        v_end = section.v_bars * units_per_bar
        f_end = v_end + section.f_bars * units_per_bar
        e_end = f_end + section.e_bars * units_per_bar

        total_units = e_end

        if not terminals:
            # Still produce a cadence at end if possible
            cad_dur = int(motif_dur_units[0])
            cad_start = max(0, total_units - cad_dur)
            return [Event(tok="CAD", start_units=cad_start, dur_units=cad_dur)]

        # duration sampling distribution
        if motif_dur_probs is None:
            motif_dur_probs = [1.0 / len(motif_dur_units)] * len(motif_dur_units)

        def sample_dur(tok: Symbol) -> int:
            if tok == "CAD":
                return int(max(motif_dur_units))
            x = rng.random()
            acc = 0.0
            for d, p in zip(motif_dur_units, motif_dur_probs):
                acc += p
                if x <= acc:
                    return int(d)
            return int(motif_dur_units[-1])

        # --- gap policy derived from density ---
        # Coverage fraction: how much of the time should be motif events (rest is infill gaps)
        coverage = 0.25 + 0.55 * density  # 0.25..0.80

        # Max gap in units: low density allows bigger gaps, high density clamps gaps
        if cfg.max_gap_units is None:
            max_gap_units = int(round((1.5 - 1.2 * density) * units_per_bar))  # ~12..2 units
            max_gap_units = max(1, max_gap_units)
        else:
            max_gap_units = max(1, int(cfg.max_gap_units))

        min_gap_units = max(0, int(cfg.min_gap_units))

        # Split terminals into V/F/E chunks (simple proportional split)
        v_units = v_end
        f_units = f_end - v_end
        e_units = e_end - f_end

        n = len(terminals)
        v_n = max(1, round(n * (v_units / total_units)))
        f_n = max(1, round(n * (f_units / total_units)))
        v_n = min(v_n, n)
        f_n = min(f_n, n - v_n)
        e_n = max(0, n - v_n - f_n)

        v_toks = list(terminals[:v_n])
        f_toks = list(terminals[v_n : v_n + f_n])
        e_toks = list(terminals[v_n + f_n : v_n + f_n + e_n])

        # Ensure CAD only appears once at the end (remove any accidental CADs)
        v_toks = [t for t in v_toks if t != "CAD"]
        f_toks = [t for t in f_toks if t != "CAD"]
        e_toks = [t for t in e_toks if t != "CAD"]

        # Reserve cadence at end
        cad_dur = sample_dur("CAD")
        cad_start = max(0, total_units - cad_dur)

        events: List[Event] = []

        def emit_with_gaps(
            start_t: int,
            end_t: int,
            toks: List[Symbol],
            *,
            motif_units_budget: int,
            allow_until_idx: Optional[int] = None,
        ) -> None:
            """
            Place events from toks with bounded gaps until motif_units_budget is used
            or toks are exhausted.
            """
            nonlocal events
            t = start_t
            used = 0
            i = 0
            while i < len(toks) and t < end_t and used < motif_units_budget:
                if allow_until_idx is not None and i >= allow_until_idx:
                    break
                tok = toks[i]
                d = sample_dur(tok)
                if d <= 0:
                    i += 1
                    continue
                if t + d > end_t:
                    break
                events.append(Event(tok=tok, start_units=t, dur_units=d))
                used += d
                t += d
                i += 1

                if t >= end_t or used >= motif_units_budget:
                    break

                # sample a gap size (small gaps more likely when density is high)
                gap_cap = min(max_gap_units, end_t - t)
                if gap_cap <= 0:
                    break

                # with higher density, bias strongly toward small gaps
                p_small = 0.65 + 0.30 * density  # 0.65..0.95
                if gap_cap <= min_gap_units:
                    gap = gap_cap
                else:
                    if rng.random() < p_small:
                        gap = min_gap_units
                    else:
                        gap = rng.randint(min_gap_units, gap_cap)

                t += gap

        # Compute motif time budgets per section (leave rest to infill)
        v_budget = int(round(v_units * coverage))
        f_budget = int(round(f_units * coverage))

        # For pre-cadence window, allocate a small budget that increases with density
        pre_e_start = f_end
        pre_e_end = cad_start
        pre_e_units = max(0, pre_e_end - pre_e_start)
        pre_e_coverage = 0.10 + 0.35 * density  # 0.10..0.45
        pre_e_budget = int(round(pre_e_units * pre_e_coverage))

        # Emit V and F
        emit_with_gaps(0, v_end, v_toks, motif_units_budget=v_budget)
        emit_with_gaps(v_end, f_end, f_toks, motif_units_budget=f_budget)

        # Emit optional material between end of F and cadence start
        if pre_e_units > 0 and e_toks:
            emit_with_gaps(pre_e_start, pre_e_end, e_toks, motif_units_budget=pre_e_budget)

        # Force cadence at end
        events.append(Event(tok="CAD", start_units=cad_start, dur_units=cad_dur))

        # Final guard: cap total number of events (prevents pathological outputs)
        if len(events) > cfg.max_events_per_section:
            events = events[: cfg.max_events_per_section]
            # Ensure CAD remains last
            if events[-1].tok != "CAD":
                events[-1] = Event(tok="CAD", start_units=cad_start, dur_units=cad_dur)

        events.sort(key=lambda e: e.start_units)
        return events

    def sample_plan(
        self,
        *,
        num_bars: int,
        section: Optional[SectionSpec] = None,
        cfg: Optional[SamplerConfig] = None,
        rng: Optional[random.Random] = None,
        motif_dur_units: Sequence[int] = (4,),
        motif_dur_probs: Optional[Sequence[float]] = None,
    ) -> Sequence[Event]:
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


def make_minimal_mvp_grammar(density: float = 0.5) -> PCFG:
    """
    Small stable grammar. density in [0,1] adjusts SPIN recursion probability.
    """
    density = max(0.0, min(1.0, density))

    base = {
        "SEQ+1": 0.18,
        "SEQ+2": 0.18,
        "INV": 0.16,
        "RET": 0.16,
        "M0": 0.32,
    }

    # recursion probability increases with density
    p_rec = 0.10 + density * (0.45 - 0.10)  # 0.10..0.45
    p_rest = 1.0 - p_rec

    base_sum = sum(base.values())
    scaled = {k: (v / base_sum) * p_rest for k, v in base.items()}

    rules = [
        GrammarRule("S", ("V", "F", "E"), 1.0),

        GrammarRule("V", ("M0", "REP"), 0.6),
        GrammarRule("V", ("M0",), 0.4),

        GrammarRule("F", ("SPIN",), 1.0),

        GrammarRule("SPIN", ("SPIN", "SPIN"), p_rec),
        GrammarRule("SPIN", ("SEQ+1", "SEQ+2"), scaled["SEQ+1"]),
        GrammarRule("SPIN", ("SEQ+2", "SEQ+1"), scaled["SEQ+2"]),
        GrammarRule("SPIN", ("INV",), scaled["INV"]),
        GrammarRule("SPIN", ("RET",), scaled["RET"]),
        GrammarRule("SPIN", ("INV",), scaled["M0"]),

        GrammarRule("E", ("CAD",), 1.0),
    ]
    return PCFG(rules=rules, start_symbol="S")


if __name__ == "__main__":
    grammar = make_minimal_mvp_grammar(density=0.7)
    cfg = SamplerConfig(seed=42, density=0.7, min_gap_units=0)
    plan = grammar.sample_plan(num_bars=8, cfg=cfg)
    for e in plan:
        print(e)