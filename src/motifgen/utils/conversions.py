import music21 as m21
import numpy as np
from typing import List, Dict

def score_to_midi_pitches(score: m21.stream.Score, min_time: int | None=None, max_time: int | None=None) -> Dict[str, List[int]]:
    part_pitches = {}
    for i, part in enumerate(score.parts):
        midi_pitches = []
        for n in part.flat:
            # Check time constraints
            offset = n.offset
            if min_time is not None and offset < min_time:
                continue
            if max_time is not None and offset > max_time:
                continue

            # Process notes and rests
            if isinstance(n, m21.note.Note):
                pitch = n.pitch.midi
            elif isinstance(n, m21.note.Rest):
                pitch = 0
            else:
                continue
            midi_pitches.append(pitch)
        part_name = part.partName if part.partName else f"Part_{i+1}"
        part_pitches[part_name] = np.asarray(midi_pitches)
    return part_pitches

def midi_to_scale_degrees(midi_pitches: List[int], key: m21.key.Key) -> List[int]:
    """Convert a list of MIDI pitches to scale degrees in the given key."""
    scale_degrees = []
    for midi in midi_pitches:
        if midi == 0:
            scale_degrees.append(0)  # Represent rests as 0
        else:
            pitch = m21.pitch.Pitch(midi=midi)
            degree = key.getScaleDegreeFromPitch(pitch)
            scale_degrees.append(degree)
    return scale_degrees

def scale_degrees_to_midi(scale_degrees: List[int], key: m21.key.Key) -> List[int]:
    """Convert a list of scale degrees in the given key to their MIDI pitches."""
    midi_pitches = []
    for scale_degree in scale_degrees:
        if scale_degree == 0:
            midi_pitches.append(0)  # Represent rests as 0
        else:
            pitch = key.pitchFromDegree(scale_degree)
            midi_pitches.append(pitch)
    return midi_pitches