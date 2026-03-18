# src/motifgen/accompaniment.py
# accompaniment style planning + rendering from RN plan

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import random
import re

import music21 as m21

AccompStyle = Literal["BLOCK", "ALBERTI", "ARP_SLOW", "ARP_FAST"]
StyleMode = Literal["single", "by_section"]


_RN_ROOT_RE = re.compile(r"^([ivIV]+o?)")


@dataclass(frozen=True)
class AccompConfig:
    """
    Rendering config.

    Assumptions:
      - rn_plan is HALF-BAR resolution: len = 2 * num_bars
      - Each RN lasts 2 beats => quarterLength = 2.0
    """
    units_per_beat: int = 2
    beats_per_bar: int = 4

    bass_octave: int = 2
    lo: str = "C3"
    hi: str = "C4"

    # rhythmic density inside each half-bar (2 beats):
    # - ALBERTI uses 4 notes by default (quavers)
    # - ARP_SLOW uses 2 notes (crotchets)
    # - ARP_FAST uses 8 notes
    arp_fast_notes_per_halfbar: int = 4

    part_name: str = "Accompaniment"


@dataclass(frozen=True)
class SectionSpec:
    """
    Same shape as pcfg.SectionSpec, duplicated here to avoid circular import.
    Bars must sum to num_bars.
    """
    v_bars: int
    f_bars: int
    e_bars: int


# ----------------------------
# Style planning
# ----------------------------

def _weighted_choice(rng: random.Random, items: Sequence[Tuple[AccompStyle, float]]) -> AccompStyle:
    s = sum(w for _, w in items)
    x = rng.random() * s
    acc = 0.0
    for k, w in items:
        acc += w
        if x <= acc:
            return k
    return items[-1][0]


def sample_style_plan(
    *,
    num_bars: int,
    section: Optional[SectionSpec],
    mode: StyleMode,
    seed: int,
) -> List[AccompStyle]:
    """
    Return a half-bar style plan (len = 2*num_bars).

    mode:
      - "single": one style for entire piece
      - "by_section": choose style per V/F/E, then expand; changes only at borders.
    """
    rng = random.Random(seed)
    n_halfbars = 2 * num_bars

    if mode == "single":
        # overall distribution: slightly conservative by default
        style = _weighted_choice(
            rng,
            [
                ("BLOCK", 0.35),
                ("ALBERTI", 0.35),
                ("ARP_SLOW", 0.20),
                ("ARP_FAST", 0.10),
            ],
        )
        return [style] * n_halfbars

    # by_section
    if section is None:
        # safe default: 2 / (num_bars-3) / 1
        v = max(1, min(2, num_bars - 2))
        e = 1
        f = max(1, num_bars - v - e)
        section = SectionSpec(v_bars=v, f_bars=f, e_bars=e)

    if section.v_bars + section.f_bars + section.e_bars != num_bars:
        raise ValueError("SectionSpec must sum to num_bars")

    # Fortspinnung more likely to be rhythmically active
    v_style = _weighted_choice(rng, [("BLOCK", 0.55), ("ALBERTI", 0.35), ("ARP_SLOW", 0.10), ("ARP_FAST", 0.00)])
    f_style = _weighted_choice(rng, [("BLOCK", 0.15), ("ALBERTI", 0.35), ("ARP_SLOW", 0.15), ("ARP_FAST", 0.35)])
    e_style = _weighted_choice(rng, [("BLOCK", 0.55), ("ALBERTI", 0.25), ("ARP_SLOW", 0.20), ("ARP_FAST", 0.00)])

    plan: List[AccompStyle] = []
    plan.extend([v_style] * (2 * section.v_bars))
    plan.extend([f_style] * (2 * section.f_bars))
    plan.extend([e_style] * (2 * section.e_bars))

    # pad/trim defensively
    plan = plan[:n_halfbars]
    while len(plan) < n_halfbars:
        plan.append(e_style)

    return plan


# ----------------------------
# RN → pitches utilities
# ----------------------------

def _rn_norm(rn: str) -> str:
    return rn.replace("°", "o").replace("ø", "o")


def _rn_root(rn: str) -> str:
    if not rn:
        return ""
    s = rn.strip().replace("°", "o").replace("ø", "o").split("/")[0]
    m = _RN_ROOT_RE.match(s)
    return m.group(1) if m else ""


def _nearest_pitch_with_pc(target: m21.pitch.Pitch, pc: int, lo_midi: int, hi_midi: int) -> m21.pitch.Pitch:
    best: Optional[m21.pitch.Pitch] = None
    best_dist = 1e18
    base_oct = target.octave if target.octave is not None else 4

    for octv in range(base_oct - 3, base_oct + 4):
        p = m21.pitch.Pitch()
        p.pitchClass = pc
        p.octave = octv
        if not (lo_midi <= p.midi <= hi_midi):
            continue
        d = abs(p.midi - target.midi)
        if d < best_dist:
            best, best_dist = p, d

    if best is None:
        p = m21.pitch.Pitch()
        p.pitchClass = pc
        p.midi = min(max(target.midi, lo_midi), hi_midi)
        while p.pitchClass != pc and p.midi < hi_midi:
            p.midi += 1
        while p.pitchClass != pc and p.midi > lo_midi:
            p.midi -= 1
        best = p

    return best


def _voicelead_upper(prev_uppers: List[m21.pitch.Pitch], chord_pcs: List[int], lo_midi: int, hi_midi: int) -> List[m21.pitch.Pitch]:
    remaining = chord_pcs[:]
    new: List[m21.pitch.Pitch] = []
    for v in prev_uppers:
        best_i = 0
        best_p = _nearest_pitch_with_pc(v, remaining[0], lo_midi, hi_midi)
        best_d = abs(best_p.midi - v.midi)
        for i, pc in enumerate(remaining[1:], start=1):
            cand = _nearest_pitch_with_pc(v, pc, lo_midi, hi_midi)
            d = abs(cand.midi - v.midi)
            if d < best_d:
                best_i, best_p, best_d = i, cand, d
        new.append(best_p)
        remaining.pop(best_i)
    new.sort(key=lambda p: p.midi)
    return new


def _voicing_left_hand(
    rn_obj: m21.roman.RomanNumeral,
    *,
    bass_octave: int,
    prev_chord: Optional[m21.chord.Chord],
    lo: str,
    hi: str,
) -> m21.chord.Chord:
    """
    Similar to realise.py: bass + 2/3 upper voices with simple voice-leading.
    """
    if not rn_obj.pitches:
        return m21.chord.Chord([])

    lo_midi = m21.pitch.Pitch(lo).midi
    hi_midi = m21.pitch.Pitch(hi).midi

    bass = rn_obj.bass()
    bass_p = m21.pitch.Pitch(bass.name)
    bass_p.octave = bass_octave

    pcs: List[int] = []
    for p in rn_obj.pitches:
        pc = int(p.pitchClass)
        if pc not in pcs:
            pcs.append(pc)

    bass_pc = bass_p.pitchClass
    upper_pcs = [pc for pc in pcs if pc != bass_pc] or [bass_pc]

    want = 3 if len(pcs) >= 4 else 2
    while len(upper_pcs) < want:
        upper_pcs.append(upper_pcs[-1])
    upper_pcs = upper_pcs[:want]

    if prev_chord is not None and len(prev_chord.pitches) >= 2:
        prev = sorted(prev_chord.pitches, key=lambda p: p.midi)[1:]
        prev = prev[-want:] if len(prev) >= want else prev
        while len(prev) < want:
            prev.insert(0, prev[0])
        uppers = _voicelead_upper(prev, upper_pcs, lo_midi, hi_midi)
    else:
        uppers: List[m21.pitch.Pitch] = []
        last = bass_p
        for pc in upper_pcs:
            p = _nearest_pitch_with_pc(last, pc, lo_midi, hi_midi)
            while p.midi <= last.midi and p.midi < hi_midi:
                p.midi += 12
            uppers.append(p)
            last = p
        uppers.sort(key=lambda p: p.midi)

    return m21.chord.Chord([bass_p] + uppers)


# ----------------------------
# Rendering patterns
# ----------------------------

def _add_rn_label(el: m21.base.Music21Object, label: str) -> None:
    try:
        el.addLyric(label)
    except Exception:
        pass


def _chord_tones_ordered(ch: m21.chord.Chord) -> List[m21.pitch.Pitch]:
    ps = list(ch.pitches)
    ps.sort(key=lambda p: p.midi)
    return ps


def _render_block(ch: m21.chord.Chord, *, dur_ql: float) -> List[m21.base.Music21Object]:
    ch2 = m21.chord.Chord(list(ch.pitches))
    ch2.duration.quarterLength = dur_ql
    return [ch2]


def _render_alberti(ch: m21.chord.Chord, *, dur_ql: float) -> List[m21.base.Music21Object]:
    # 4 notes in half-bar: (1-5-3-5) using pitches from voiced chord
    ps = _chord_tones_ordered(ch)
    if len(ps) == 0:
        return [m21.note.Rest(quarterLength=dur_ql)]
    bass = ps[0]
    fifth = ps[-1]  # highest pitch (often 5th or 7th)
    third = ps[1] if len(ps) > 1 else ps[0]
    pat = [bass, fifth, third, fifth]
    q_each = dur_ql / 4.0
    return [m21.note.Note(p, quarterLength=q_each) for p in pat]


def _render_arp_slow(ch: m21.chord.Chord, *, dur_ql: float) -> List[m21.base.Music21Object]:
    # 2 notes per half-bar: bass then upper
    ps = _chord_tones_ordered(ch)
    if len(ps) == 0:
        return [m21.note.Rest(quarterLength=dur_ql)]
    bass = ps[0]
    upper = ps[-1] if len(ps) > 1 else ps[0]
    q_each = dur_ql / 2.0
    return [m21.note.Note(bass, quarterLength=q_each), m21.note.Note(upper, quarterLength=q_each)]


def _render_arp_fast(ch: m21.chord.Chord, *, dur_ql: float, n_notes: int) -> List[m21.base.Music21Object]:
    ps = _chord_tones_ordered(ch)
    if len(ps) == 0:
        return [m21.note.Rest(quarterLength=dur_ql)]
    # cycle through chord tones upward
    pat = []
    for i in range(n_notes):
        pat.append(ps[i % len(ps)])
    q_each = dur_ql / float(n_notes)
    return [m21.note.Note(p, quarterLength=q_each) for p in pat]


def render_style(
    style: AccompStyle,
    *,
    chord: m21.chord.Chord,
    dur_ql: float,
    cfg: AccompConfig,
) -> List[m21.base.Music21Object]:
    if style == "BLOCK":
        return _render_block(chord, dur_ql=dur_ql)
    if style == "ALBERTI":
        return _render_alberti(chord, dur_ql=dur_ql)
    if style == "ARP_SLOW":
        return _render_arp_slow(chord, dur_ql=dur_ql)
    if style == "ARP_FAST":
        return _render_arp_fast(chord, dur_ql=dur_ql, n_notes=cfg.arp_fast_notes_per_halfbar)
    return _render_block(chord, dur_ql=dur_ql)


# ----------------------------
# Public API: realise accompaniment part
# ----------------------------

def realise_accompaniment_part(
    *,
    key_obj: m21.key.Key,
    rn_plan: Sequence[str],
    style_plan: Sequence[AccompStyle],
    cfg: AccompConfig,
) -> m21.stream.Part:
    """
    Render a chordal accompaniment from:
      - rn_plan: half-bar roman numerals (len = 2*num_bars)
      - style_plan: half-bar styles (same length as rn_plan)
    Produces a music21 Part with patterns.
    """
    # harden key object (avoid accidental C major defaults)
    try:
        key_obj = m21.key.Key(key_obj.tonic.name, key_obj.mode)
    except Exception:
        pass

    n = min(len(rn_plan), len(style_plan))
    part = m21.stream.Part()
    part.partName = cfg.part_name
    part.insert(0.0, key_obj)

    halfbar_units = 2 * cfg.units_per_beat
    dur_ql = halfbar_units / float(cfg.units_per_beat)  # == 2.0

    prev_chord: Optional[m21.chord.Chord] = None

    for i in range(n):
        rn = (rn_plan[i] or "N").strip()
        style = style_plan[i]
        off_ql = (i * halfbar_units) / float(cfg.units_per_beat)

        if rn == "N":
            r = m21.note.Rest(quarterLength=dur_ql)
            _add_rn_label(r, "N")
            part.insert(off_ql, r)
            prev_chord = None
            continue

        rn_norm = _rn_norm(rn)
        try:
            rn_obj = m21.roman.RomanNumeral(rn_norm, key_obj)
            chord = _voicing_left_hand(
                rn_obj,
                bass_octave=cfg.bass_octave,
                prev_chord=prev_chord,
                lo=cfg.lo,
                hi=cfg.hi,
            )
            # label the first element of the pattern with the RN for readability
            pattern = render_style(style, chord=chord, dur_ql=dur_ql, cfg=cfg)
            if pattern:
                _add_rn_label(pattern[0], rn_norm)

            # place pattern sequentially inside the half-bar window
            local_off = off_ql
            for el in pattern:
                part.insert(local_off, el)
                local_off += float(el.duration.quarterLength)

            prev_chord = chord
        except Exception:
            r = m21.note.Rest(quarterLength=dur_ql)
            _add_rn_label(r, "N")
            part.insert(off_ql, r)
            prev_chord = None

    return part