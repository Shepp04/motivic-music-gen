# music21 extraction + tokenisation + splits

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import music21 as m21


BAR = "BAR"


def load_corpus_ids(limit: int | None) -> list[str]:
    """
    Return parseable Bach chorale identifiers using the official chorales iterator.

    Per music21 docs, Iterator(returnType='filename') yields strings like:
        'bach/bwv269', 'bach/bwv347', ...
    which can be passed to music21.corpus.parse().  [oai_citation:1‡music21.org](https://music21.org/music21docs/moduleReference/moduleCorpusChorales.html?utm_source=chatgpt.com)
    """
    it = m21.corpus.chorales.Iterator(returnType="filename")
    ids: list[str] = []

    for chorale_id in it:
        # chorale_id is a string like 'bach/bwv269'
        if isinstance(chorale_id, str) and chorale_id:
            ids.append(chorale_id)

        if limit is not None and len(ids) >= limit:
            break

    return ids


def process_piece(corpus_id: str, target_key: str | None = None) -> dict[str, Any]:
    """
    Parse a piece, analyse its (original) key, extract melody + harmony.
    target_key is ignored (kept only so you don't have to change call sites).
    """
    score = m21.corpus.parse(corpus_id)

    try:
        key_obj = score.analyze("key")
    except Exception:
        key_obj = m21.key.Key("C")

    melody_part = extract_melody(score)
    melody_tokens = tokenise_melody(melody_part, key_obj=key_obj)

    harmony_tokens = extract_harmony(score, key_obj=key_obj)

    return {
        "id": corpus_id,
        "key": str(key_obj),
        "melody_tokens": melody_tokens,
        "harmony_tokens": harmony_tokens,
    }


def extract_melody(score) -> m21.stream.Part:
    """
    MVP: for chorales, take the top part as "melody".
    Fallback: choose the part with the highest average pitch.
    """
    parts = list(score.parts)
    if not parts:
        raise ValueError("Score has no parts")

    # Heuristic 1: chorales often have soprano as part 0
    if len(parts) >= 4:
        return parts[0]

    # Heuristic 2: choose part with highest average MIDI pitch over notes
    def avg_pitch(part: m21.stream.Part) -> float:
        ps = [n.pitch.midi for n in part.recurse().notes if n.pitch is not None]
        if not ps:
            return -1e9
        return sum(ps) / len(ps)

    best = max(parts, key=avg_pitch)
    return best


def _snap_duration(q_len: float, dur_set: tuple[float, ...]) -> float:
    """
    Snap a duration (in quarterLength units) to nearest allowed duration.
    """
    return min(dur_set, key=lambda d: abs(d - q_len))


def _pc_from_pitch(p: m21.pitch.Pitch, key_obj: m21.key.Key) -> int:
    """
    Return tonic-relative pitch class in [0..11].
    0 = tonic, 1 = tonic+1 semitone, ... 11 = tonic-1 semitone.
    """
    tonic_pc = key_obj.tonic.pitchClass
    return (p.pitchClass - tonic_pc) % 12


def get_anacrusis_shift(score_or_part) -> float:
    """
    If the first measure is incomplete, return its duration (quarterLength)
    so we can align time 0 to the first full downbeat.
    """
    measured = score_or_part.makeMeasures(inPlace=False)
    measures = list(measured.getElementsByClass(m21.stream.Measure))
    if not measures:
        return 0.0

    m0 = measures[0]

    ts = m0.timeSignature
    bar_len = float(ts.barDuration.quarterLength) if ts is not None else 4.0

    # Actual filled duration of the first measure
    m0_dur = float(m0.duration.quarterLength)

    if 1e-6 < m0_dur < bar_len - 1e-6:
        return m0_dur
    return 0.0


def tokenise_melody(
    part: m21.stream.Part,
    key_obj: m21.key.Key,
    dur_set=(2.0, 1.0, 0.5),
) -> list[str]:
    """
    Tokenise into:
      - BAR
      - N:<pc>:<dur>  (pc is tonic-relative pitch class 0..11)
      - R:<dur>

    Handles:
      - anacrusis (drops first incomplete measure)
      - non-4/4 meters (uses bar length from time signature)
      - padding each kept bar to its true bar length
    """
    measured = part.makeMeasures(inPlace=False)
    shift = get_anacrusis_shift(part)

    tokens: list[str] = []
    measures = list(measured.getElementsByClass(m21.stream.Measure))

    for meas_idx, meas in enumerate(measures):
        # Drop pickup measure if present
        if meas_idx == 0 and shift > 1e-6:
            continue

        ts = meas.timeSignature
        bar_len = float(ts.barDuration.quarterLength) if ts is not None else 4.0

        tokens.append(BAR)

        remaining = bar_len
        elems = list(meas.notesAndRests)

        for el in elems:
            q = float(el.duration.quarterLength)

            while q > 1e-9 and remaining > 1e-9:
                chunk = min(q, remaining)
                q -= chunk
                remaining -= chunk

                snapped = _snap_duration(chunk, dur_set)

                if isinstance(el, m21.note.Rest):
                    tokens.append(f"R:{snapped}")
                elif isinstance(el, m21.note.Note):
                    pc = _pc_from_pitch(el.pitch, key_obj)
                    tokens.append(f"N:{pc}:{snapped}")
                elif isinstance(el, m21.chord.Chord):
                    pitch = el.sortAscending().pitches[-1]
                    pc = _pc_from_pitch(pitch, key_obj)
                    tokens.append(f"N:{pc}:{snapped}")
                else:
                    tokens.append(f"R:{snapped}")

                if remaining <= 1e-9:
                    break

            if remaining <= 1e-9:
                break

        # Pad to fill the bar deterministically
        while remaining > 1e-9:
            pad = min(remaining, max(dur_set))
            snapped = _snap_duration(pad, dur_set)
            tokens.append(f"R:{snapped}")
            remaining -= snapped

    return tokens


def extract_harmony(score: m21.stream.Score, key_obj: m21.key.Key) -> list[str]:
    """
    Roman numeral labels at crotchet (quarter-note) resolution.
    Index 0 corresponds to the first full-bar downbeat (pickup removed).
    """
    chordified = score.chordify()
    shift = get_anacrusis_shift(score)

    total_q = int(max(0.0, (float(chordified.highestTime) - shift) // 1.0))

    labels: list[str] = []
    flat = chordified.flat

    for i in range(total_q):
        t = shift + float(i)

        cs = flat.getElementsByOffset(
            t,
            t + 1.0,
            mustBeginInSpan=False,
            includeEndBoundary=False,
            classList=[m21.chord.Chord],
        )

        if not cs:
            labels.append("N")
            continue

        c = cs[0]
        try:
            rn = m21.roman.romanNumeralFromChord(c, key_obj)
            labels.append(str(rn.figure))
        except Exception:
            labels.append("N")

    return labels


def train_val_test_split(items: list[Any], seed: int = 0) -> tuple[list[Any], list[Any], list[Any]]:
    """
    80/10/10 split with deterministic shuffle.
    """
    rng = random.Random(seed)
    items_shuf = items[:]
    rng.shuffle(items_shuf)

    n = len(items_shuf)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = items_shuf[:n_train]
    val = items_shuf[n_train : n_train + n_val]
    test = items_shuf[n_train + n_val :]
    return train, val, test


def save_jsonl(items: list[dict[str, Any]], path: str | Path) -> None:
    """
    Save list of dicts to JSONL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Example usage: process n chorales and save to JSONL
    corpus_ids = load_corpus_ids(limit=100) #['bach/bwv66.6.xml'] 
    data = [process_piece(cid, target_key="C") for cid in corpus_ids]
    train, val, test = train_val_test_split(data, seed=42)
    save_jsonl(train, "data/train.jsonl")
    save_jsonl(val, "data/val.jsonl")
    save_jsonl(test, "data/test.jsonl")