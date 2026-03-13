# src/motifgen/realise.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import re

import music21 as m21

Token = str
Symbol = str

_RN_ROOT_RE = re.compile(r"^([ivIV]+o?)")


# ----------------------------
# Data models
# ----------------------------

@dataclass(frozen=True)
class Event:
    tok: Symbol
    start_units: int
    dur_units: int


@dataclass(frozen=True)
class MotifEvent:
    deg: int          # 0..6 diatonic scale degree
    oct: int          # octave number
    dur_units: int
    is_rest: bool = False


Motif = List[MotifEvent]


# ----------------------------
# Motif stream <-> diatonic motif
# ----------------------------

def motif_from_stream(stream: m21.stream.Stream, *, key_obj: m21.key.Key, units_per_beat: int) -> Motif:
    out: Motif = []
    for el in stream.notesAndRests:
        du = max(1, int(round(float(el.duration.quarterLength) * units_per_beat)))

        if isinstance(el, m21.note.Rest):
            out.append(MotifEvent(0, 4, du, True))
            continue

        if isinstance(el, m21.note.Note):
            pitch = el.pitch
        elif isinstance(el, m21.chord.Chord):
            pitch = el.sortAscending().pitches[-1]
        else:
            out.append(MotifEvent(0, 4, du, True))
            continue

        deg_1 = key_obj.getScaleDegreeFromPitch(pitch)
        if deg_1 is None:
            deg_1 = key_obj.getScale().getScaleDegreeFromPitch(pitch) or 1

        out.append(MotifEvent((int(deg_1) - 1) % 7, int(pitch.octave), du, False))

    return out


def pitch_from_deg_oct(deg: int, octv: int, key_obj: m21.key.Key) -> m21.pitch.Pitch:
    p = key_obj.pitchFromDegree((deg % 7) + 1)
    p.octave = octv
    return p


def motif_to_stream(motif: Motif, *, key_obj: m21.key.Key, units_per_beat: int, color: Optional[str] = None) -> m21.stream.Stream:
    s = m21.stream.Stream()
    for ev in motif:
        ql = ev.dur_units / float(units_per_beat)
        el = m21.note.Rest(quarterLength=ql) if ev.is_rest else m21.note.Note(pitch_from_deg_oct(ev.deg, ev.oct, key_obj), quarterLength=ql)
        if color:
            try:
                el.style.color = color
            except Exception:
                pass
        s.append(el)
    return s


# ----------------------------
# Motif transforms
# ----------------------------

def diatonic_shift(motif: Motif, n: int) -> Motif:
    out: Motif = []
    for ev in motif:
        if ev.is_rest:
            out.append(ev)
            continue
        nd = ev.deg + n
        oct_shift = nd // 7 if nd >= 0 else -((-nd + 6) // 7)
        out.append(MotifEvent(nd % 7, ev.oct + oct_shift, ev.dur_units, False))
    return out


def inv(motif: Motif, axis_deg: Optional[int] = None) -> Motif:
    if axis_deg is None:
        axis_deg = next((e.deg for e in motif if not e.is_rest), None)
        if axis_deg is None:
            return list(motif)

    out: Motif = []
    for ev in motif:
        if ev.is_rest:
            out.append(ev)
            continue
        d = 2 * axis_deg - ev.deg
        oct_shift = d // 7 if d >= 0 else -((-d + 6) // 7)
        out.append(MotifEvent(d % 7, ev.oct + oct_shift, ev.dur_units, False))
    return out


def apply_motif_token(base: Motif, tok: str) -> Motif:
    if tok in ("M0", "REP"):
        return list(base)
    if tok == "RET":
        return list(reversed(base))
    if tok == "INV":
        return inv(base)
    if tok.startswith("SEQ"):
        return diatonic_shift(base, int(tok.replace("SEQ", "")))
    return list(base)


def fit_motif_to_duration(motif: Motif, target_units: int) -> Motif:
    """
    Safe fit: ensures sum(dur_units)==target_units and NEVER loops forever.

    Policy:
      - If target_units is smaller than the minimum possible (len(motif_nonzero)),
        truncate the motif to fit (keep earliest events).
      - Otherwise do your scale+round, then adjust with bounded passes.
    """
    if target_units <= 0:
        return []

    # Drop empty motif
    if not motif:
        return [MotifEvent(deg=0, oct=4, dur_units=target_units, is_rest=True)]

    # Minimum possible duration if each event must be >=1 unit
    min_units = len(motif)
    if target_units < min_units:
        # Truncate to exactly target_units events of 1 unit each.
        # Keep first (target_units) events; if you prefer “keep shape”, keep first events.
        trimmed = motif[:target_units]
        return [MotifEvent(ev.deg, ev.oct, 1, ev.is_rest) for ev in trimmed]

    total = sum(ev.dur_units for ev in motif)
    if total <= 0:
        return [MotifEvent(deg=0, oct=4, dur_units=target_units, is_rest=True)]

    # Scale and round (>=1)
    scaled: List[MotifEvent] = []
    for ev in motif:
        new_d = max(1, int(round(ev.dur_units * (target_units / total))))
        scaled.append(MotifEvent(ev.deg, ev.oct, new_d, ev.is_rest))

    # Fix to exact target with a bounded adjustment loop
    cur = sum(ev.dur_units for ev in scaled)
    diff = target_units - cur

    # First, if we need to subtract, only subtract from events with dur>1
    if diff < 0:
        i = 0
        guard = 0
        while diff < 0 and guard < 10_000:
            ev = scaled[i % len(scaled)]
            if ev.dur_units > 1:
                scaled[i % len(scaled)] = MotifEvent(ev.deg, ev.oct, ev.dur_units - 1, ev.is_rest)
                diff += 1
            i += 1
            guard += 1

        # If still negative, we hit the floor (all 1s). Truncate to fit.
        if diff < 0:
            # Keep first target_units events, all 1 unit
            return [MotifEvent(ev.deg, ev.oct, 1, ev.is_rest) for ev in scaled[:target_units]]

    # If we need to add, distribute +1s
    if diff > 0:
        i = 0
        while diff > 0:
            ev = scaled[i % len(scaled)]
            scaled[i % len(scaled)] = MotifEvent(ev.deg, ev.oct, ev.dur_units + 1, ev.is_rest)
            diff -= 1
            i += 1

    return scaled


def make_cadence_template(*, key_obj: m21.key.Key, units_per_beat: int, dur_units: int, octave: int = 4) -> Motif:
    half = max(1, dur_units // 2)
    return [
        MotifEvent(1, octave, half, False),
        MotifEvent(0, octave, max(1, dur_units - half), False),
    ]


# ----------------------------
# Harmony-aware motif snapping + tokenisation
# ----------------------------

def _rn_root_is_V(rn: str) -> bool:
    if not rn or rn == "N":
        return False
    s = rn.strip().replace("°", "o").replace("ø", "o").split("/")[0]
    m = _RN_ROOT_RE.match(s)
    return bool(m and m.group(1) == "V")


def _strong_onset(start_units: int, *, units_per_beat: int, beats_per_bar: int) -> bool:
    # only count note onsets that land exactly on beat 1 or 3
    if start_units % units_per_beat != 0:
        return False
    beat = (start_units // units_per_beat) % beats_per_bar
    return beat in (0, 2)


def _rn_chord_pcs(rn: str, key_obj: m21.key.Key) -> List[int]:
    if not rn or rn == "N":
        return []
    rn_norm = rn.replace("°", "o").replace("ø", "o")
    try:
        rn_obj = m21.roman.RomanNumeral(rn_norm, key_obj)
        pcs: List[int] = []
        for p in rn_obj.pitches:
            pc = int(p.pitchClass)
            if pc not in pcs:
                pcs.append(pc)
        return pcs
    except Exception:
        return []


def _deg_to_pc_abs(deg: int, key_obj: m21.key.Key) -> int:
    return int(key_obj.pitchFromDegree((deg % 7) + 1).pitchClass)


def _closest_diatonic_degree_to_pc(target_pc_abs: int, chord_pcs_abs: Sequence[int], key_obj: m21.key.Key) -> int:
    # choose nearest chord pc to the degree pc, then map that pc back to one of 0..6
    if not chord_pcs_abs:
        return 0
    def dist(a: int, b: int) -> int:
        d = abs(a - b) % 12
        return min(d, 12 - d)

    best_pc = chord_pcs_abs[0]
    best_d = dist(target_pc_abs, best_pc)
    for pc in chord_pcs_abs[1:]:
        d = dist(target_pc_abs, pc)
        if d < best_d:
            best_pc, best_d = pc, d

    for d in range(7):
        if _deg_to_pc_abs(d, key_obj) == best_pc:
            return d
    return 0


def harmonise_motif_block_to_rn(
    motif: Motif,
    *,
    key_obj: m21.key.Key,
    rn_plan: Sequence[str],
    block_start_units: int,
    units_per_beat: int,
    beats_per_bar: int,
) -> Motif:
    if not rn_plan:
        return motif

    out: Motif = []
    t = block_start_units
    halfbar_units = 2 * units_per_beat

    for ev in motif:
        if ev.is_rest:
            out.append(ev)
            t += ev.dur_units
            continue

        new_deg = ev.deg
        if _strong_onset(t, units_per_beat=units_per_beat, beats_per_bar=beats_per_bar):
            hi = t // halfbar_units
            rn = rn_plan[hi] if 0 <= hi < len(rn_plan) else "N"
            pcs = _rn_chord_pcs(rn, key_obj)
            if pcs:
                new_deg = _closest_diatonic_degree_to_pc(_deg_to_pc_abs(ev.deg, key_obj), pcs, key_obj)

        out.append(MotifEvent(new_deg, ev.oct, ev.dur_units, False))
        t += ev.dur_units

    return out


def motif_block_to_tokens_harmony_aware(
    motif: Motif,
    *,
    key_obj: m21.key.Key,
    units_per_beat: int,
    rn_plan: Optional[Sequence[str]],
    block_start_units: int,
) -> List[Token]:
    """
    Convert motif block into tokens (pc-relative to tonic).
    Extra rule: in minor, during V harmony, pc 10 -> pc 11 (raised leading tone).
    """
    tonic_pc = key_obj.tonic.pitchClass
    is_minor = (key_obj.mode or "").lower() == "minor"
    halfbar_units = 2 * units_per_beat

    out: List[Token] = []
    t = block_start_units

    for idx, ev in enumerate(motif):
        ql = float(ev.dur_units / float(units_per_beat))

        if ev.is_rest:
            out.append(f"R:{ql}")
            t += ev.dur_units
            continue

        p = pitch_from_deg_oct(ev.deg, ev.oct, key_obj)
        rel_pc = (p.pitchClass - tonic_pc) % 12

        if rn_plan and is_minor:
            hi = t // halfbar_units
            rn = rn_plan[hi] if 0 <= hi < len(rn_plan) else "N"
            if _rn_root_is_V(rn):
                if rel_pc == 10:
                    rel_pc = 11
                if rel_pc == 8:
                    prev_note_pc = next((int(out[j].split(":")[1]) for j in range(len(out)-1, -1, -1) if out[j].startswith("N:")), None)
                    next_note_pc = None
                    for k in range(idx + 1, len(motif)):
                        if motif[k].is_rest:
                            continue
                        p2 = pitch_from_deg_oct(motif[k].deg, motif[k].oct, key_obj)
                        next_note_pc = (p2.pitchClass - tonic_pc) % 12
                        if next_note_pc == 10:
                            next_note_pc = 11  # because we would raise it
                        break

                    if (prev_note_pc in (11, 10)) or (next_note_pc in (11, 10)):
                        rel_pc = 9

        out.append(f"N:{int(rel_pc)}:{ql}")
        t += ev.dur_units

    return out


def motif_events_to_token_map(
    *,
    key_obj: m21.key.Key,
    units_per_beat: int,
    base_motif: Motif,
    events: Sequence[Event],
    rn_plan: Optional[Sequence[str]] = None,
    beats_per_bar: int = 4,
) -> Dict[Tuple[int, int], List[Token]]:
    out: Dict[Tuple[int, int], List[Token]] = {}
    for ev in events:
        if ev.dur_units <= 0:
            continue

        if ev.tok == "CAD":
            block = make_cadence_template(key_obj=key_obj, units_per_beat=units_per_beat, dur_units=ev.dur_units, octave=4)
        else:
            block = fit_motif_to_duration(apply_motif_token(base_motif, ev.tok), ev.dur_units)

        if rn_plan is not None:
            block = harmonise_motif_block_to_rn(
                block,
                key_obj=key_obj,
                rn_plan=rn_plan,
                block_start_units=ev.start_units,
                units_per_beat=units_per_beat,
                beats_per_bar=beats_per_bar,
            )

        out[(ev.start_units, ev.dur_units)] = motif_block_to_tokens_harmony_aware(
            block,
            key_obj=key_obj,
            units_per_beat=units_per_beat,
            rn_plan=rn_plan,
            block_start_units=ev.start_units,
        )
    return out


# ----------------------------
# Token rendering (melody part)
# ----------------------------

def _choose_pitch_nearby(target_pc: int, prev: Optional[m21.pitch.Pitch], default_octave: int, max_oct_off: int = 1) -> m21.pitch.Pitch:
    # candidates at default_octave +/- 1; pick minimal leap + slight drift penalty
    cands: List[m21.pitch.Pitch] = []
    for off in range(-max_oct_off, max_oct_off + 1):
        p = m21.pitch.Pitch()
        p.pitchClass = target_pc
        p.octave = default_octave + off
        cands.append(p)

    if prev is None:
        return cands[max_oct_off]  # off=0

    best = cands[0]
    best_cost = 1e18
    for p in cands:
        leap = abs(p.midi - prev.midi)
        drift = abs(p.octave - default_octave)
        cost = leap + 2.0 * drift
        if cost < best_cost:
            best_cost = cost
            best = p
    return best


def tokens_to_part(
    tokens: Sequence[Token],
    *,
    key_obj: m21.key.Key,
    default_octave: int = 4,
    color_spans: Optional[List[Tuple[int, int, str]]] = None,
) -> m21.stream.Part:
    part = m21.stream.Part()
    part.insert(0.0, key_obj)

    color_for: Dict[int, str] = {}
    if color_spans:
        for a, b, c in color_spans:
            for i in range(a, b):
                color_for[i] = c

    tonic_pc = key_obj.tonic.pitchClass
    prev: Optional[m21.pitch.Pitch] = None

    for i, tok in enumerate(tokens):
        if tok.startswith("R:"):
            ql = float(tok.split(":")[1])
            part.append(m21.note.Rest(quarterLength=ql))
            prev = None
            continue

        if tok.startswith("N:"):
            _, pc_s, dur_s = tok.split(":")
            rel_pc = int(pc_s)
            ql = float(dur_s)

            target_pc = (tonic_pc + rel_pc) % 12
            p = _choose_pitch_nearby(target_pc, prev, default_octave, max_oct_off=1)
            n = m21.note.Note(p, quarterLength=ql)

            col = color_for.get(i)
            if col:
                try:
                    n.style.color = col
                except Exception:
                    pass

            part.append(n)
            prev = p

    return part


# ----------------------------
# Harmony realisation (kept largely as-is)
# ----------------------------

def _add_rn_label(el: m21.base.Music21Object, label: str) -> None:
    try:
        el.addLyric(label)
    except Exception:
        pass


def _rn_norm(rn: str) -> str:
    return rn.replace("°", "o").replace("ø", "o")


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
    bass_octave: int = 2,
    prev_chord: Optional[m21.chord.Chord] = None,
    lo: str = "C3",
    hi: str = "C4",
) -> m21.chord.Chord:
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


def realise_harmony_part(
    *,
    key_obj: m21.key.Key,
    rn_plan: Sequence[str],
    num_bars: int,
    units_per_beat: int = 4,
    beats_per_bar: int = 4,
    bass_octave: int = 2,
    part_name: str = "Harmony",
) -> m21.stream.Part:
    units_per_bar = beats_per_bar * units_per_beat
    total_units = num_bars * units_per_bar
    halfbar_units = 2 * units_per_beat
    expected_len = total_units // halfbar_units

    rns = list(rn_plan[:expected_len])
    while len(rns) < expected_len:
        rns.append("N")

    part = m21.stream.Part()
    part.partName = part_name
    part.insert(0.0, key_obj)

    prev: Optional[m21.chord.Chord] = None

    for i, rn in enumerate(rns):
        off_ql = (i * halfbar_units) / float(units_per_beat)
        dur_ql = halfbar_units / float(units_per_beat)

        rn = (rn or "N").strip()
        if rn == "N":
            r = m21.note.Rest(quarterLength=dur_ql)
            _add_rn_label(r, "N")
            part.insert(off_ql, r)
            prev = None
            continue

        rn_norm = _rn_norm(rn)
        try:
            rn_obj = m21.roman.RomanNumeral(rn_norm, key_obj)
            ch = _voicing_left_hand(rn_obj, bass_octave=bass_octave, prev_chord=prev)
            ch.duration.quarterLength = dur_ql
            _add_rn_label(ch, rn_norm)
            part.insert(off_ql, ch)
            prev = ch
        except Exception:
            r = m21.note.Rest(quarterLength=dur_ql)
            _add_rn_label(r, "N")
            part.insert(off_ql, r)
            prev = None

    return part


# ----------------------------
# High-level assembly (kept for backward compat)
# ----------------------------

def realise_piece(
    *,
    key_obj: m21.key.Key,
    motif_stream: m21.stream.Stream,
    events: Sequence[Event],
    num_bars: int,
    units_per_beat: int = 4,
    beats_per_bar: int = 4,
    rn_plan: Optional[Sequence[str]] = None,
    color_map: Optional[Dict[str, str]] = None,
) -> m21.stream.Part:
    # This is only used for quick debugging renders. The main pipeline uses:
    # motif_events_to_token_map + melody_ngram.infill_timeline_with_spans + tokens_to_part
    if color_map is None:
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

    units_per_bar = beats_per_bar * units_per_beat
    total_units = num_bars * units_per_bar
    base = motif_from_stream(motif_stream, key_obj=key_obj, units_per_beat=units_per_beat)

    part = m21.stream.Part()
    part.insert(0.0, key_obj)

    occupied = [False] * total_units
    placed: List[Tuple[int, Motif, Optional[str]]] = []

    for ev in sorted(events, key=lambda e: e.start_units):
        if ev.start_units < 0 or ev.dur_units <= 0 or ev.start_units >= total_units:
            continue
        start = ev.start_units
        end = min(total_units, start + ev.dur_units)
        if any(occupied[t] for t in range(start, end)):
            continue

        if ev.tok == "CAD":
            block = make_cadence_template(key_obj=key_obj, units_per_beat=units_per_beat, dur_units=end - start, octave=4)
        else:
            block = fit_motif_to_duration(apply_motif_token(base, ev.tok), end - start)

        if rn_plan is not None:
            block = harmonise_motif_block_to_rn(
                block,
                key_obj=key_obj,
                rn_plan=rn_plan,
                block_start_units=start,
                units_per_beat=units_per_beat,
                beats_per_bar=beats_per_bar,
            )

        for t in range(start, end):
            occupied[t] = True
        placed.append((start, block, color_map.get(ev.tok)))

    cur = 0
    for start, block, col in sorted(placed, key=lambda x: x[0]):
        if start > cur:
            part.append(m21.note.Rest(quarterLength=(start - cur) / float(units_per_beat)))
            cur = start

        for el in motif_to_stream(block, key_obj=key_obj, units_per_beat=units_per_beat, color=col).notesAndRests:
            part.append(el)
        cur += sum(e.dur_units for e in block)

    if cur < total_units:
        part.append(m21.note.Rest(quarterLength=(total_units - cur) / float(units_per_beat)))

    return part


def realise_score(
    *,
    key_obj: m21.key.Key,
    motif_stream: m21.stream.Stream,
    events: Sequence[Event],
    rn_plan: Sequence[str],
    num_bars: int,
    units_per_beat: int = 4,
    beats_per_bar: int = 4,
) -> m21.stream.Score:
    mel = realise_piece(
        key_obj=key_obj,
        motif_stream=motif_stream,
        events=events,
        rn_plan=rn_plan,
        num_bars=num_bars,
        units_per_beat=units_per_beat,
        beats_per_bar=beats_per_bar,
    )
    harm = realise_harmony_part(
        key_obj=key_obj,
        rn_plan=rn_plan,
        num_bars=num_bars,
        units_per_beat=units_per_beat,
        beats_per_bar=beats_per_bar,
        bass_octave=2,
    )
    sc = m21.stream.Score()
    sc.insert(0.0, mel)
    sc.insert(0.0, harm)
    return sc