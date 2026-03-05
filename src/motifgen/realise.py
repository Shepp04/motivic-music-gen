# motif transforms + realise tokens into notes
# src/motifgen/realise.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import music21 as m21

Symbol = str  # nonterminal or terminal

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

# ----------------------------
# Motif representation
# ----------------------------

@dataclass(frozen=True)
class MotifEvent:
    deg: int            # 0..6 (tonic-relative diatonic degree)
    oct: int            # octave number (e.g. 4)
    dur_units: int      # duration in grid units (units_per_beat = 2 => half-beat)
    is_rest: bool = False


Motif = List[MotifEvent]


# ----------------------------
# Utilities: stream <-> motif
# ----------------------------

def motif_from_stream(
    stream: m21.stream.Stream,
    *,
    key_obj: m21.key.Key,
    units_per_beat: int,
) -> Motif:
    """
    Convert a music21 motif stream into a key-relative diatonic Motif.

    - scale degree is 0..6 relative to key tonic
    - durations quantised onto the units grid
    """
    evs: Motif = []
    for el in stream.notesAndRests:
        ql = float(el.duration.quarterLength)
        # 1 quarterLength = 1 beat (in music21)
        dur_units = max(1, int(round(ql * units_per_beat)))

        if isinstance(el, m21.note.Rest):
            evs.append(MotifEvent(deg=0, oct=4, dur_units=dur_units, is_rest=True))
            continue

        if isinstance(el, m21.note.Note):
            pitch = el.pitch
        elif isinstance(el, m21.chord.Chord):
            pitch = el.sortAscending().pitches[-1]
        else:
            evs.append(MotifEvent(deg=0, oct=4, dur_units=dur_units, is_rest=True))
            continue

        deg_1based = key_obj.getScaleDegreeFromPitch(pitch)
        if deg_1based is None:
            # If chromatic, snap to nearest scale degree
            deg_1based = key_obj.getScale().getScaleDegreeFromPitch(pitch) or 1

        deg = int(deg_1based) - 1
        evs.append(MotifEvent(deg=deg % 7, oct=int(pitch.octave), dur_units=dur_units, is_rest=False))

    return evs


def pitch_from_deg_oct(deg: int, octv: int, key_obj: m21.key.Key) -> m21.pitch.Pitch:
    p = key_obj.pitchFromDegree((deg % 7) + 1)  # 1..7
    p.octave = octv
    return p


def motif_to_stream(
    motif: Motif,
    *,
    key_obj: m21.key.Key,
    units_per_beat: int,
    color: Optional[str] = None,
) -> m21.stream.Stream:
    s = m21.stream.Stream()
    for ev in motif:
        ql = ev.dur_units / float(units_per_beat)
        if ev.is_rest:
            n = m21.note.Rest(quarterLength=ql)
        else:
            p = pitch_from_deg_oct(ev.deg, ev.oct, key_obj)
            n = m21.note.Note(p, quarterLength=ql)

        if color is not None:
            # music21 supports style.color for many renderers
            try:
                n.style.color = color
            except Exception:
                pass

        s.append(n)
    return s


# ----------------------------
# Diatonic transformations
# ----------------------------

def diatonic_shift(motif: Motif, n: int) -> Motif:
    out: Motif = []
    for ev in motif:
        if ev.is_rest:
            out.append(ev)
            continue
        new_deg = ev.deg + n
        oct_shift = new_deg // 7 if new_deg >= 0 else -((-new_deg + 6) // 7)
        out.append(MotifEvent(deg=new_deg % 7, oct=ev.oct + oct_shift, dur_units=ev.dur_units))
    return out


def ret(motif: Motif) -> Motif:
    return list(reversed(motif))


def inv(motif: Motif, axis_deg: Optional[int] = None) -> Motif:
    if axis_deg is None:
        for ev in motif:
            if not ev.is_rest:
                axis_deg = ev.deg
                break
        if axis_deg is None:
            return list(motif)

    out: Motif = []
    for ev in motif:
        if ev.is_rest:
            out.append(ev)
            continue
        inv_deg = 2 * axis_deg - ev.deg
        oct_shift = inv_deg // 7 if inv_deg >= 0 else -((-inv_deg + 6) // 7)
        out.append(MotifEvent(deg=inv_deg % 7, oct=ev.oct + oct_shift, dur_units=ev.dur_units))
    return out


def apply_motif_token(base: Motif, tok: str) -> Motif:
    if tok in ("M0", "REP"):
        return list(base)
    if tok == "RET":
        return ret(base)
    if tok == "INV":
        return inv(base)
    if tok.startswith("SEQ"):
        # format: "SEQ+2" or "SEQ-1"
        n = int(tok.replace("SEQ", ""))
        return diatonic_shift(base, n)
    # CAD handled separately (cadence template)
    return list(base)

# Time scaling and fitting

def fit_motif_to_duration(motif: Motif, target_units: int) -> Motif:
    """
    Stretch/compress motif durations so total dur_units sums to target_units.
    Keeps relative durations approximately.
    """
    if target_units <= 0:
        return []

    total = sum(ev.dur_units for ev in motif)
    if total <= 0:
        return [MotifEvent(deg=0, oct=4, dur_units=target_units, is_rest=True)]

    # Scale and round, then fix rounding error
    scaled = []
    for ev in motif:
        new_d = max(1, int(round(ev.dur_units * (target_units / total))))
        scaled.append(MotifEvent(deg=ev.deg, oct=ev.oct, dur_units=new_d, is_rest=ev.is_rest))

    # Adjust to exact target_units
    diff = target_units - sum(ev.dur_units for ev in scaled)
    i = 0
    while diff != 0 and scaled:
        ev = scaled[i % len(scaled)]
        if diff > 0:
            scaled[i % len(scaled)] = MotifEvent(ev.deg, ev.oct, ev.dur_units + 1, ev.is_rest)
            diff -= 1
        else:
            if ev.dur_units > 1:
                scaled[i % len(scaled)] = MotifEvent(ev.deg, ev.oct, ev.dur_units - 1, ev.is_rest)
                diff += 1
        i += 1

    return scaled

# Harmony

def _add_rn_label(el: m21.base.Music21Object, label: str) -> None:
    """
    Attach a printable label to a chord/rest for score display.
    Uses lyrics because they're reliably rendered by music21.
    """
    try:
        el.addLyric(label)
    except Exception:
        pass


def _rn_normalise_dim_symbol(rn: str) -> str:
    """
    Ensure diminished symbol is in the format music21 accepts.
    We normalise to 'o' (e.g. 'viio7', 'viio42'). If input uses '°'/'ø', convert.
    """
    return rn.replace("°", "o").replace("ø", "o")


def _nearest_pitch_with_pc(
    target: m21.pitch.Pitch,
    pc: int,
    lo_midi: int,
    hi_midi: int,
) -> m21.pitch.Pitch:
    """
    Return a pitch with pitchClass=pc that is closest to target (in semitones),
    constrained to [lo_midi, hi_midi].
    """
    best: Optional[m21.pitch.Pitch] = None
    best_dist = 1e9

    # Search octaves around target
    base_oct = target.octave if target.octave is not None else 4
    for octv in range(base_oct - 3, base_oct + 4):
        p = m21.pitch.Pitch()
        p.pitchClass = pc
        p.octave = octv
        midi = p.midi
        if midi < lo_midi or midi > hi_midi:
            continue
        dist = abs(midi - target.midi)
        if dist < best_dist:
            best = p
            best_dist = dist

    # Fallback: clamp target to range and set pc in that octave
    if best is None:
        p = m21.pitch.Pitch()
        p.pitchClass = pc
        # choose closest octave boundary
        midi_target = min(max(target.midi, lo_midi), hi_midi)
        p.midi = midi_target
        # fix pitchClass by moving within octave
        while p.pitchClass != pc and p.midi < hi_midi:
            p.midi += 1
        while p.pitchClass != pc and p.midi > lo_midi:
            p.midi -= 1
        best = p

    return best


def _voicelead_upper(
    prev_uppers: List[m21.pitch.Pitch],
    chord_pcs: List[int],
    lo_midi: int,
    hi_midi: int,
) -> List[m21.pitch.Pitch]:
    """
    Choose upper voices (same count as prev_uppers) that minimise movement.
    Greedy matching: for each previous voice, pick the nearest available chord tone.
    """
    remaining_pcs = chord_pcs[:]
    new_uppers: List[m21.pitch.Pitch] = []

    for v in prev_uppers:
        # pick best pc for this voice
        best_i = 0
        best_p = _nearest_pitch_with_pc(v, remaining_pcs[0], lo_midi, hi_midi)
        best_dist = abs(best_p.midi - v.midi)

        for i, pc in enumerate(remaining_pcs[1:], start=1):
            cand = _nearest_pitch_with_pc(v, pc, lo_midi, hi_midi)
            dist = abs(cand.midi - v.midi)
            if dist < best_dist:
                best_dist = dist
                best_i = i
                best_p = cand

        new_uppers.append(best_p)
        remaining_pcs.pop(best_i)

    # Sort uppers to avoid voice crossing (low->high)
    new_uppers.sort(key=lambda p: p.midi)
    return new_uppers


def _voicing_left_hand(
    rn_obj: m21.roman.RomanNumeral,
    bass_octave: int = 2,
    prev_chord: Optional[m21.chord.Chord] = None,
    lo: str = "C3",
    hi: str = "C4",
) -> m21.chord.Chord:
    """
    Left-hand voicing with simple voice-leading.

    - Bass = rn_obj.bass() at bass_octave (respects inversion)
    - Upper voices chosen to minimise movement from prev_chord (if provided)
    - Range limited to [lo, hi]
    """
    chord_pitches = list(rn_obj.pitches)
    if not chord_pitches:
        return m21.chord.Chord([])

    lo_midi = m21.pitch.Pitch(lo).midi
    hi_midi = m21.pitch.Pitch(hi).midi

    # Bass pitch (fixed by inversion)
    bass = rn_obj.bass()
    bass_p = m21.pitch.Pitch(bass.name)  # ignore octave from rn_obj
    bass_p.octave = bass_octave

    # Chord pitch classes (unique)
    pcs = []
    for p in chord_pitches:
        pc = p.pitchClass
        if pc not in pcs:
            pcs.append(pc)

    # Ensure bass pc is included and remove it from upper selection pool
    bass_pc = bass_p.pitchClass
    upper_pcs = [pc for pc in pcs if pc != bass_pc]
    if not upper_pcs:
        upper_pcs = [bass_pc]  # degenerate chord (won't happen often)

    # Choose how many upper voices: 2 (triad-ish) or 3 (7th-ish / dense)
    want_uppers = 3 if len(pcs) >= 4 else 2
    # If we don't have enough pcs (e.g. 2 unique pcs), allow duplicates
    while len(upper_pcs) < want_uppers:
        upper_pcs.append(upper_pcs[-1])

    upper_pcs = upper_pcs[:want_uppers]

    # If we have a previous chord, voice-lead against its upper voices
    if prev_chord is not None and len(prev_chord.pitches) >= 2:
        prev_pitches = list(prev_chord.pitches)
        prev_pitches.sort(key=lambda p: p.midi)
        prev_uppers = prev_pitches[1:]  # exclude bass
        # Match count
        prev_uppers = prev_uppers[-want_uppers:] if len(prev_uppers) >= want_uppers else prev_uppers
        while len(prev_uppers) < want_uppers:
            prev_uppers.insert(0, prev_uppers[0])
        uppers = _voicelead_upper(prev_uppers, upper_pcs, lo_midi, hi_midi)
    else:
        # No previous: stack above bass, tightly, within range
        uppers = []
        last = bass_p
        for pc in upper_pcs:
            p = _nearest_pitch_with_pc(last, pc, lo_midi, hi_midi)
            while p.midi <= last.midi and p.midi < hi_midi:
                p.midi += 12
            uppers.append(p)
            last = p
        uppers.sort(key=lambda p: p.midi)

    voiced = [bass_p] + uppers
    ch = m21.chord.Chord(voiced)
    return ch


def make_cadence_template(
    *,
    key_obj: m21.key.Key,
    units_per_beat: int,
    dur_units: int,
    octave: int = 4,
) -> Motif:
    """
    Simple diatonic cadence cell ending on tonic:
      scale degree 1 -> 0 (i.e., 2 -> 1 in 1-based)
    Fitted to dur_units.
    """
    # 0-based: 1 = supertonic, 0 = tonic
    half = max(1, dur_units // 2)
    base = [
        MotifEvent(deg=1, oct=octave, dur_units=half, is_rest=False),
        MotifEvent(deg=0, oct=octave, dur_units=max(1, dur_units - half), is_rest=False),
    ]
    return base


# Placement and assembly
def realise_harmony_part(
    *,
    key_obj: m21.key.Key,
    rn_plan: Sequence[str],
    num_bars: int,
    units_per_beat: int = 2,
    beats_per_bar: int = 4,
    bass_octave: int = 3,
    part_name: str = "Harmony",
) -> m21.stream.Part:
    """
    Realise a half-bar Roman numeral plan into a chordal accompaniment Part.

    Assumptions:
      - rn_plan is HALF-BAR resolution: length should be 2 * num_bars
      - Each RN lasts 2 beats => quarterLength = 2.0
      - "N" indicates no harmony (rest)

    Returns a music21 Part with Chord/Rest events placed at offsets.
    """
    units_per_bar = beats_per_bar * units_per_beat
    total_units = num_bars * units_per_bar

    # Each half-bar is 2 beats => 2 * units_per_beat units
    halfbar_units = 2 * units_per_beat
    expected_len = (total_units // halfbar_units)

    # Clamp/trim/pad rn_plan safely (MVP)
    rns = list(rn_plan[:expected_len])
    while len(rns) < expected_len:
        rns.append("N")

    part = m21.stream.Part()
    part.partName = part_name
    part.insert(0.0, key_obj)

    prev_chord: Optional[m21.chord.Chord] = None
    
    for i, rn in enumerate(rns):
        offset_units = i * halfbar_units
        offset_ql = offset_units / float(units_per_beat)
        dur_ql = halfbar_units / float(units_per_beat)  # == 2.0

        rn = rn.strip() if rn else "N"

        if rn == "N":
            r = m21.note.Rest(quarterLength=dur_ql)
            _add_rn_label(r, "N")
            part.insert(offset_ql, r)
            continue

        rn_norm = _rn_normalise_dim_symbol(rn)

        try:
            rn_obj = m21.roman.RomanNumeral(rn_norm, key_obj)
            ch = _voicing_left_hand(rn_obj, bass_octave=bass_octave, prev_chord=prev_chord)
            ch.duration.quarterLength = dur_ql

            # Label with the (normalised) RN figure you planned
            _add_rn_label(ch, rn_norm)

            part.insert(offset_ql, ch)
            prev_chord = ch
        except Exception:
            r = m21.note.Rest(quarterLength=dur_ql)
            _add_rn_label(r, "N")
            part.insert(offset_ql, r)
            prev_chord = None

    return part


def realise_piece(
    *,
    key_obj: m21.key.Key,
    motif_stream: m21.stream.Stream,
    events: Sequence[Event],
    num_bars: int,
    units_per_beat: int = 2,
    beats_per_bar: int = 4,
    rn_plan: Optional[Sequence[str]] = None,  # accepted for future harmony-aware constraints
    color_map: Optional[Dict[str, str]] = None,
) -> m21.stream.Part:
    """
    Build a single monophonic music21 Part by:
      1) converting user motif to internal diatonic representation
      2) applying PCFG structure tokens (M0/SEQ/INV/RET/REP/CAD)
      3) fitting each realised motif to the event dur_units
      4) placing motif material onto a half-beat grid timeline
      5) filling uncovered gaps with rests (Markov infill can replace this later)
      6) returning a music21 Part

    Collision policy (MVP): if an event overlaps already-filled time, it is skipped.
    """
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

    # Convert motif to internal diatonic motif
    base_motif = motif_from_stream(motif_stream, key_obj=key_obj, units_per_beat=units_per_beat)

    # Timeline as a list of (token, is_rest, deg, oct) at unit granularity is overkill;
    # we'll place as time-sorted note/rest events into a Part.
    # For collision checks, track occupied units.
    occupied = [False] * total_units

    # Sort events by time
    events_sorted = sorted(events, key=lambda e: e.start_units)

    # Collect placed motif fragments as (start_units, motif_fragment, color)
    placed: List[Tuple[int, Motif, Optional[str]]] = []

    for ev in events_sorted:
        if ev.start_units < 0 or ev.dur_units <= 0:
            continue
        if ev.start_units >= total_units:
            continue

        start = ev.start_units
        end = min(total_units, start + ev.dur_units)
        dur = end - start
        if dur <= 0:
            continue

        # Skip if overlaps already-filled time
        if any(occupied[t] for t in range(start, end)):
            continue

        # Realise motif token
        if ev.tok == "CAD":
            realised = make_cadence_template(
                key_obj=key_obj,
                units_per_beat=units_per_beat,
                dur_units=dur,
                octave=4,
            )
        else:
            realised = apply_motif_token(base_motif, ev.tok)
            realised = fit_motif_to_duration(realised, dur)

        # Mark occupied
        for t in range(start, end):
            occupied[t] = True

        placed.append((start, realised, color_map.get(ev.tok)))

    # Assemble into a single Part
    part = m21.stream.Part()
    part.insert(0.0, key_obj)

    # Walk through time and emit rests + motif blocks
    placed.sort(key=lambda x: x[0])
    cur = 0

    for start, motif_block, color in placed:
        if start > cur:
            # gap -> rest (Markov infill can go here later)
            gap_units = start - cur
            gap_ql = gap_units / float(units_per_beat)
            part.append(m21.note.Rest(quarterLength=gap_ql))
            cur = start

        # Emit motif notes/rests
        block_stream = motif_to_stream(motif_block, key_obj=key_obj, units_per_beat=units_per_beat, color=color)
        for el in block_stream.notesAndRests:
            part.append(el)

        cur += sum(ev.dur_units for ev in motif_block)

    # Tail rest
    if cur < total_units:
        tail_ql = (total_units - cur) / float(units_per_beat)
        part.append(m21.note.Rest(quarterLength=tail_ql))

    return part


def realise_score(
    *,
    key_obj: m21.key.Key,
    motif_stream: m21.stream.Stream,
    events: Sequence[Event],
    rn_plan: Sequence[str],
    num_bars: int,
    units_per_beat: int = 2,
    beats_per_bar: int = 4,
) -> m21.stream.Score:
    melody_part = realise_piece(
        key_obj=key_obj,
        motif_stream=motif_stream,
        events=events,
        rn_plan=rn_plan,
        num_bars=num_bars,
        units_per_beat=units_per_beat,
        beats_per_bar=beats_per_bar,
    )
    harmony_part = realise_harmony_part(
        key_obj=key_obj,
        rn_plan=rn_plan,
        num_bars=num_bars,
        units_per_beat=units_per_beat,
        beats_per_bar=beats_per_bar,
        bass_octave=2,
    )

    sc = m21.stream.Score()
    sc.insert(0.0, melody_part)
    sc.insert(0.0, harmony_part)
    return sc


if __name__ == "__main__":
    # Example usage
    key = m21.key.Key('C')
    motif_stream = m21.stream.Stream()
    motif_stream.append(m21.note.Note('C4', quarterLength=0.5))
    motif_stream.append(m21.note.Note('D4', quarterLength=0.5))
    motif_stream.append(m21.note.Note('E4', quarterLength=1.0))

    events = [
        Event(tok="M0", start_units=0, dur_units=8),
        Event(tok="SEQ+2", start_units=8, dur_units=8),
        Event(tok="INV", start_units=16, dur_units=8),
        Event(tok="CAD", start_units=24, dur_units=8),
    ]

    rn_plan = ["I", "vi", "IV", "V", "I63", "vi", "V", "I"]  # Example harmony plan for 4 bars

    score = realise_score(
        key_obj=key,
        motif_stream=motif_stream,
        events=events,
        rn_plan=rn_plan,
        num_bars=4,
        units_per_beat=2,
        beats_per_bar=4,
    )

    score.show()