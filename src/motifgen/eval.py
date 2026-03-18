# src/motifgen/eval.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math

import music21 as m21

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Symbol = str

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

Token = str


_NOTE_PREFIX = "N:"
_REST_PREFIX = "R:"
_BAR = "BAR"


# ----------------------------
# Config / results
# ----------------------------

@dataclass(frozen=True)
class EvalConfig:
    beats_per_bar: int = 4
    units_per_beat: int = 4
    # strong beats: beats 1 and 3 (0-indexed -> 0,2)
    strong_beats: Tuple[int, ...] = (0, 2)
    # For self-similarity segmentation
    segment: str = "bar"  # "bar" | "halfbar"
    # For SSM feature vector
    use_pc_hist: bool = True
    use_rhythm_hist: bool = True


@dataclass(frozen=True)
class CorpusStats:
    # interval stats computed over corpus
    step_rate: float
    leap7_rate: float
    mean_abs_interval: float
    p95_abs_interval: float
    dur_hist_bins: Tuple[float, ...]
    dur_hist_probs: Tuple[float, ...]


# ----------------------------
# Token parsing / timing helpers
# ----------------------------

def is_note(tok: Token) -> bool:
    return tok.startswith(_NOTE_PREFIX)


def is_rest(tok: Token) -> bool:
    return tok.startswith(_REST_PREFIX)


def parse_note(tok: Token) -> Tuple[int, float]:
    # N:<pc>:<dur>
    _, pc_s, dur_s = tok.split(":")
    return int(pc_s), float(dur_s)


def parse_rest(tok: Token) -> float:
    # R:<dur>
    return float(tok.split(":")[1])


def token_quarter_length(tok: Token) -> float:
    if is_note(tok):
        return float(tok.split(":")[2])
    if is_rest(tok):
        return parse_rest(tok)
    raise ValueError(f"Unknown token: {tok}")


def token_units(tok: Token, *, units_per_beat: int) -> int:
    ql = token_quarter_length(tok)  # music21: 1.0 == 1 beat
    u = int(round(ql * units_per_beat))
    return max(1, u)


def tokens_strip_bar(tokens: Sequence[Token]) -> List[Token]:
    return [t for t in tokens if t != _BAR and t.strip()]


def _strong_onset(start_units: int, *, units_per_beat: int, beats_per_bar: int) -> bool:
    # user-provided definition
    if start_units % units_per_beat != 0:
        return False
    beat = (start_units // units_per_beat) % beats_per_bar
    return beat in (0, 2)


def halfbar_index_for_unit(cur_unit: int, *, units_per_beat: int) -> int:
    # half-bar = 2 beats
    return cur_unit // (2 * units_per_beat)


def bar_index_for_unit(cur_unit: int, *, units_per_beat: int, beats_per_bar: int) -> int:
    return cur_unit // (units_per_beat * beats_per_bar)


def total_units_from_num_bars(num_bars: int, *, units_per_beat: int, beats_per_bar: int) -> int:
    return num_bars * units_per_beat * beats_per_bar


# ----------------------------
# Harmony chord tones
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
# Core metric computations
# ----------------------------

def melody_onsets(tokens: Sequence[Token], *, units_per_beat: int) -> List[Tuple[int, Token]]:
    """
    Returns list of (start_units, tok) for each token, including rests.
    """
    out: List[Tuple[int, Token]] = []
    t = 0
    for tok in tokens:
        out.append((t, tok))
        t += token_units(tok, units_per_beat=units_per_beat)
    return out


def extract_note_pcs_in_time(tokens: Sequence[Token], *, units_per_beat: int) -> List[Tuple[int, int]]:
    """
    Returns [(start_units, pc)] for NOTE tokens only.
    """
    out: List[Tuple[int, int]] = []
    t = 0
    for tok in tokens:
        if is_note(tok):
            pc, _dur = parse_note(tok)
            out.append((t, pc))
        t += token_units(tok, units_per_beat=units_per_beat)
    return out


def interval_stats_from_tokens(tokens: Sequence[Token], *, units_per_beat: int) -> Dict[str, float]:
    """
    Compute interval magnitudes in semitone space using tonic-relative pcs (mod-12).
    This is coarse (octave-free) but still correlates with step/leap behaviour.
    """
    pcs: List[int] = []
    for tok in tokens:
        if is_note(tok):
            pc, _ = parse_note(tok)
            pcs.append(pc)

    if len(pcs) < 2:
        return {
            "step_rate": float("nan"),
            "leap7_rate": float("nan"),
            "mean_abs_interval": float("nan"),
            "p95_abs_interval": float("nan"),
        }

    # circular distance on pitch class ring (0..6)
    abs_ints: List[int] = []
    for a, b in zip(pcs[:-1], pcs[1:]):
        d = abs(a - b) % 12
        abs_ints.append(min(d, 12 - d))

    abs_arr = np.asarray(abs_ints, dtype=float)
    step_rate = float(np.mean(abs_arr <= 2.0))
    leap7_rate = float(np.mean(abs_arr >= 7.0))
    mean_abs = float(np.mean(abs_arr))
    p95 = float(np.percentile(abs_arr, 95))

    return {
        "step_rate": step_rate,
        "leap7_rate": leap7_rate,
        "mean_abs_interval": mean_abs,
        "p95_abs_interval": p95,
    }


def strong_beat_chord_tone_rate(
    melody_tokens: Sequence[Token],
    rn_plan: Sequence[str],
    *,
    key_obj: m21.key.Key,
    units_per_beat: int,
    beats_per_bar: int,
) -> float:
    """
    SBCTR: among NOTE onsets that are strong onsets, fraction that are chord tones.
    Uses half-bar harmony.
    """
    chord_sets = chord_pc_sets_from_rn_plan(rn_plan, key_obj=key_obj)
    onsets = extract_note_pcs_in_time(melody_tokens, units_per_beat=units_per_beat)

    num = 0
    den = 0
    for start_u, pc in onsets:
        if not _strong_onset(start_u, units_per_beat=units_per_beat, beats_per_bar=beats_per_bar):
            continue
        hi = halfbar_index_for_unit(start_u, units_per_beat=units_per_beat)
        chord_pcs = chord_sets[hi] if 0 <= hi < len(chord_sets) else set()
        if not chord_pcs:
            continue  # no harmony info => skip (don’t penalise)
        den += 1
        if pc in chord_pcs:
            num += 1

    return float(num / den) if den > 0 else float("nan")


def motif_coverage(events: Sequence[Event], total_units: int) -> float:
    if total_units <= 0:
        return float("nan")

    covered_units = np.zeros(total_units, dtype=bool)
    for e in events:
        start = max(0, int(e.start_units))
        end = min(total_units, start + max(0, int(e.dur_units)))
        if end > start:
            covered_units[start:end] = True
    return float(np.mean(covered_units))


def motif_dispersion(events: Sequence[Event], *, num_bars: int, units_per_beat: int, beats_per_bar: int) -> float:
    """
    Variance of motif instances per bar (lower variance = more evenly spread).
    """
    if num_bars <= 0:
        return float("nan")
    counts = np.zeros(num_bars, dtype=float)
    for e in events:
        b = bar_index_for_unit(e.start_units, units_per_beat=units_per_beat, beats_per_bar=beats_per_bar)
        if 0 <= b < num_bars and e.tok != "CAD":
            counts[b] += 1.0
    return float(np.var(counts))


def cadence_alignment_pass(
    melody_tokens: Sequence[Token],
    rn_plan: Sequence[str],
    *,
    key_obj: m21.key.Key,
    total_units: int,
    units_per_beat: int,
    beats_per_bar: int,
) -> int:
    """
    Simple cadence pass:
      - piece ends exactly at total_units (melody duration)
      - last RN is tonic function-ish: I or i (starts with I/i)
      - last strong onset note (within final bar) is chord tone of last half-bar
    """
    # duration exact
    dur = sum(token_units(t, units_per_beat=units_per_beat) for t in melody_tokens)
    if dur != total_units:
        return 0

    # harmony ending check
    last_rn = (rn_plan[-1] if rn_plan else "N").strip()
    if not last_rn or last_rn == "N":
        return 0
    if not (last_rn.startswith("I") or last_rn.startswith("i")):
        return 0

    chord_sets = chord_pc_sets_from_rn_plan(rn_plan, key_obj=key_obj)

    # find last strong-onset note
    note_onsets = extract_note_pcs_in_time(melody_tokens, units_per_beat=units_per_beat)
    if not note_onsets:
        return 0

    last_pc = None
    last_u = None
    for u, pc in note_onsets:
        if _strong_onset(u, units_per_beat=units_per_beat, beats_per_bar=beats_per_bar):
            last_pc = pc
            last_u = u
    if last_pc is None or last_u is None:
        return 0

    hi = halfbar_index_for_unit(last_u, units_per_beat=units_per_beat)
    chord_pcs = chord_sets[hi] if 0 <= hi < len(chord_sets) else set()
    if not chord_pcs:
        return 0
    return 1 if last_pc in chord_pcs else 0


def _segment_vectors(
    melody_tokens: Sequence[Token],
    *,
    cfg: EvalConfig,
) -> np.ndarray:
    """
    Convert melody into segment feature vectors (bar or halfbar).
    Features:
      - pitch-class histogram (12)
      - rhythm histogram over durations in cfg units (based on observed token ql)
    """
    units_per_seg = (cfg.units_per_beat * cfg.beats_per_bar) if cfg.segment == "bar" else (2 * cfg.units_per_beat)
    total_u = sum(token_units(t, units_per_beat=cfg.units_per_beat) for t in melody_tokens)
    n_seg = max(1, int(math.ceil(total_u / units_per_seg)))

    # gather dur values seen (for rhythm histogram bins)
    durs = sorted({token_quarter_length(t) for t in melody_tokens if t != _BAR})
    if not durs:
        durs = [0.25, 0.5, 1.0, 2.0]

    # allocate
    vecs: List[np.ndarray] = []

    # iterate segments; build histograms by token onsets falling into seg
    onsets = melody_onsets(melody_tokens, units_per_beat=cfg.units_per_beat)

    for seg_idx in range(n_seg):
        seg_start = seg_idx * units_per_seg
        seg_end = seg_start + units_per_seg

        pc_hist = np.zeros(12, dtype=float)
        dur_hist = np.zeros(len(durs), dtype=float)

        for u, tok in onsets:
            if not (seg_start <= u < seg_end):
                continue
            ql = token_quarter_length(tok)
            # rhythm: count any token onset (rest or note)
            # pitch: count notes only
            if cfg.use_rhythm_hist:
                j = durs.index(ql) if ql in durs else None
                if j is not None:
                    dur_hist[j] += 1.0
            if cfg.use_pc_hist and is_note(tok):
                pc, _ = parse_note(tok)
                pc_hist[pc % 12] += 1.0

        # normalise each sub-histogram
        feats: List[np.ndarray] = []
        if cfg.use_pc_hist:
            s = pc_hist.sum()
            feats.append(pc_hist / s if s > 0 else pc_hist)
        if cfg.use_rhythm_hist:
            s = dur_hist.sum()
            feats.append(dur_hist / s if s > 0 else dur_hist)

        vec = np.concatenate(feats) if feats else np.zeros(1, dtype=float)
        vecs.append(vec)

    return np.stack(vecs, axis=0)


def self_similarity_score(
    melody_tokens: Sequence[Token],
    *,
    cfg: EvalConfig,
) -> Tuple[float, np.ndarray]:
    """
    Compute SSM and a scalar structure score.
    Score = mean off-diagonal similarity excluding a band around diagonal.
    """
    X = _segment_vectors(melody_tokens, cfg=cfg)  # (S,D)
    # cosine similarity
    eps = 1e-9
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Xn = X / norms
    SSM = Xn @ Xn.T  # (S,S)

    S = SSM.shape[0]
    if S <= 1:
        return float("nan"), SSM

    # exclude diagonal and near-diagonal band
    band = 1
    vals = []
    for i in range(S):
        for j in range(S):
            if i == j:
                continue
            if abs(i - j) <= band:
                continue
            vals.append(float(SSM[i, j]))
    score = float(np.mean(vals)) if vals else float("nan")
    return score, SSM


# ----------------------------
# Corpus stats
# ----------------------------

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def compute_corpus_stats(items: Sequence[Dict[str, Any]], *, cfg: EvalConfig) -> CorpusStats:
    # Use melody tokens, strip BAR, compute pooled interval stats and duration histogram
    all_tokens: List[Token] = []
    for it in items:
        toks = tokens_strip_bar(it.get("melody_tokens", []))
        all_tokens.extend(toks)

    ints = interval_stats_from_tokens(all_tokens, units_per_beat=cfg.units_per_beat)

    # duration histogram (over quarterLength values)
    durs = [token_quarter_length(t) for t in all_tokens if t != _BAR]
    if not durs:
        bins = (0.25, 0.5, 1.0, 2.0)
        probs = tuple([1.0 / len(bins)] * len(bins))
        return CorpusStats(
            step_rate=ints["step_rate"],
            leap7_rate=ints["leap7_rate"],
            mean_abs_interval=ints["mean_abs_interval"],
            p95_abs_interval=ints["p95_abs_interval"],
            dur_hist_bins=bins,
            dur_hist_probs=probs,
        )

    uniq = sorted(set(durs))
    counts = np.array([durs.count(x) for x in uniq], dtype=float)
    probs = counts / (counts.sum() if counts.sum() > 0 else 1.0)

    return CorpusStats(
        step_rate=ints["step_rate"],
        leap7_rate=ints["leap7_rate"],
        mean_abs_interval=ints["mean_abs_interval"],
        p95_abs_interval=ints["p95_abs_interval"],
        dur_hist_bins=tuple(uniq),
        dur_hist_probs=tuple(map(float, probs)),
    )


# ----------------------------
# Piece evaluation
# ----------------------------

def evaluate_piece(
    piece: Dict[str, Any],
    *,
    cfg: EvalConfig,
    corpus: Optional[CorpusStats] = None,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Evaluate a single generated piece dict
    Returns:
      - metrics dict (flat)
      - SSM matrix (for plotting)
    """
    melody_tokens = tokens_strip_bar(piece["melody_tokens"])
    rn_plan = piece.get("rn_plan", [])
    events: Sequence[Event] = piece.get("events", [])
    num_bars = int(piece["num_bars"])
    units_per_beat = int(piece["units_per_beat"])
    beats_per_bar = int(piece["beats_per_bar"])
    mode = piece.get("mode", "major")
    system = piece.get("system", "unknown")
    seed = piece.get("seed", None)

    key_obj = m21.key.Key("a") if mode == "minor" else m21.key.Key("C")

    total_units = total_units_from_num_bars(num_bars, units_per_beat=units_per_beat, beats_per_bar=beats_per_bar)
    dur_units = sum(token_units(t, units_per_beat=units_per_beat) for t in melody_tokens)

    # metrics
    ints = interval_stats_from_tokens(melody_tokens, units_per_beat=units_per_beat)
    sbctr = strong_beat_chord_tone_rate(
        melody_tokens,
        rn_plan,
        key_obj=key_obj,
        units_per_beat=units_per_beat,
        beats_per_bar=beats_per_bar,
    )
    cov = motif_coverage(events, total_units=total_units)
    disp = motif_dispersion(events, num_bars=num_bars, units_per_beat=units_per_beat, beats_per_bar=beats_per_bar)
    cad_pass = cadence_alignment_pass(
        melody_tokens,
        rn_plan,
        key_obj=key_obj,
        total_units=total_units,
        units_per_beat=units_per_beat,
        beats_per_bar=beats_per_bar,
    )

    ssm_score, ssm = self_similarity_score(melody_tokens, cfg=cfg)

    # distances to corpus stats (if provided)
    def diff(a: float, b: float) -> float:
        if (a is None) or (b is None) or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            return float("nan")
        return float(abs(a - b))

    out: Dict[str, Any] = {
        "seed": seed,
        "system": system,
        "mode": mode,
        "num_bars": num_bars,
        "units_per_beat": units_per_beat,
        "beats_per_bar": beats_per_bar,
        "total_units_expected": total_units,
        "total_units_got": dur_units,
        "duration_ok": int(dur_units == total_units),

        "sb_chord_tone_rate": sbctr,

        "step_rate": ints["step_rate"],
        "leap7_rate": ints["leap7_rate"],
        "mean_abs_interval": ints["mean_abs_interval"],
        "p95_abs_interval": ints["p95_abs_interval"],

        "ssm_score": ssm_score,

        "motif_coverage": cov,
        "motif_dispersion_var": disp,

        "cadence_pass": cad_pass,
        "n_events": len(list(events)),
    }

    if corpus is not None:
        out.update({
            "diff_step_rate": diff(out["step_rate"], corpus.step_rate),
            "diff_leap7_rate": diff(out["leap7_rate"], corpus.leap7_rate),
            "diff_mean_abs_interval": diff(out["mean_abs_interval"], corpus.mean_abs_interval),
            "diff_p95_abs_interval": diff(out["p95_abs_interval"], corpus.p95_abs_interval),
        })

    return out, ssm


def evaluate_many(
    pieces: Sequence[Dict[str, Any]],
    *,
    cfg: EvalConfig,
    corpus: Optional[CorpusStats] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for p in pieces:
        r, _ = evaluate_piece(p, cfg=cfg, corpus=corpus)
        rows.append(r)

    df = pd.DataFrame(rows)

    # summary by system
    metrics = [
        "sb_chord_tone_rate",
        "step_rate",
        "leap7_rate",
        "mean_abs_interval",
        "p95_abs_interval",
        "ssm_score",
        "motif_coverage",
        "motif_dispersion_var",
        "cadence_pass",
        "duration_ok",
    ]
    summary = (
        df.groupby(["system", "mode"], dropna=False)[metrics]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    return df, summary


# ----------------------------
# Export helpers: Excel + plots
# ----------------------------

def export_excel(
    *,
    df_per_piece: pd.DataFrame,
    df_summary: pd.DataFrame,
    corpus: Optional[CorpusStats],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_per_piece.to_excel(w, sheet_name="per_piece", index=False)
        df_summary.to_excel(w, sheet_name="summary", index=False)
        if corpus is not None:
            df_c = pd.DataFrame([{
                "step_rate": corpus.step_rate,
                "leap7_rate": corpus.leap7_rate,
                "mean_abs_interval": corpus.mean_abs_interval,
                "p95_abs_interval": corpus.p95_abs_interval,
                "dur_bins": str(list(corpus.dur_hist_bins)),
                "dur_probs": str(list(corpus.dur_hist_probs)),
            }])
            df_c.to_excel(w, sheet_name="corpus", index=False)

    return out_path


def plot_interval_hist(
    *,
    pieces: Sequence[Dict[str, Any]],
    cfg: EvalConfig,
    out_dir: str | Path,
    title: str = "Interval magnitude histogram (pc circular distance)",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build per-system abs interval arrays
    data_by_sys: Dict[str, List[int]] = {}
    for p in pieces:
        sys = p.get("system", "unknown")
        toks = tokens_strip_bar(p["melody_tokens"])
        pcs = [parse_note(t)[0] for t in toks if is_note(t)]
        abs_ints: List[int] = []
        for a, b in zip(pcs[:-1], pcs[1:]):
            d = abs(a - b) % 12
            abs_ints.append(min(d, 12 - d))
        data_by_sys.setdefault(sys, []).extend(abs_ints)

    plt.figure()
    bins = np.arange(-0.5, 6.6, 1.0)  # 0..6
    for sys, vals in data_by_sys.items():
        if not vals:
            continue
        plt.hist(vals, bins=bins, alpha=0.5, label=sys, density=True)
    plt.title(title)
    plt.xlabel("abs interval (0..6)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "interval_hist.png", dpi=200)
    plt.close()


def plot_ssm(
    *,
    ssm: np.ndarray,
    out_path: str | Path,
    title: str = "Self-similarity matrix",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(ssm, aspect="equal", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metric_bars(
    *,
    df_per_piece: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    title: Optional[str] = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = title or metric

    # mean +/- std by system
    g = df_per_piece.groupby("system")[metric]
    means = g.mean()
    stds = g.std()

    plt.figure()
    xs = np.arange(len(means))
    plt.bar(xs, means.values, yerr=stds.values, capsize=4)
    plt.xticks(xs, means.index, rotation=20, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate_from_files(
    *,
    generated_jsonl: str | Path,
    val_jsonl: str | Path = "data/val.jsonl",
    test_jsonl: str | Path = "data/test.jsonl",
    out_dir: str | Path = "outputs/eval",
    cfg: Optional[EvalConfig] = None,
) -> None:
    """
    Convenience: evaluate a jsonl of generated pieces
    """
    cfg = cfg or EvalConfig()

    gen = load_jsonl(generated_jsonl)
    val = load_jsonl(val_jsonl)
    test = load_jsonl(test_jsonl)
    corpus = compute_corpus_stats(val + test, cfg=cfg)

    df, summary = evaluate_many(gen, cfg=cfg, corpus=corpus)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_excel(df_per_piece=df, df_summary=summary, corpus=corpus, out_path=out_dir / "results.xlsx")

    # plots
    plot_interval_hist(pieces=gen, cfg=cfg, out_dir=out_dir)
    plot_metric_bars(df_per_piece=df, metric="sb_chord_tone_rate", out_path=out_dir / "sbctr.png", title="Strong-beat chord tone rate")
    plot_metric_bars(df_per_piece=df, metric="ssm_score", out_path=out_dir / "ssm_score.png", title="SSM structure score")

    # 1 SSM example
    if gen:
        r, ssm = evaluate_piece(gen[0], cfg=cfg, corpus=corpus)
        if ssm is not None:
            plot_ssm(ssm=ssm, out_path=out_dir / "ssm_example.png", title=f"SSM example ({gen[0].get('system','unknown')})")
