# src/motifgen/harmony_ngram.py
# Function n-gram + diatonic RN realiser (mode-conditioned)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random
import re

Mode = str  # "major" | "minor"
Function = str  # "T" | "PD" | "D" | "UNK"

# Keep chord vocabulary simple/stable for consonance
_ALLOWED_FIGS = {"", "63", "7", "43", "42"}  # or just {""} for root-position triads

_DIATONIC_MAJOR = {"I", "ii", "iii", "IV", "V", "vi", "viio"}
_DIATONIC_MINOR = {"i", "iio", "III", "iv", "V", "VI", "VII", "viio"}  # harmonic-minor-friendly


# ----------------------------
# Parsing / simplification
# ----------------------------

_RN_ROOT_RE = re.compile(r"^([ivIV]+o?)")
_DIGITS_RE = re.compile(r"\d+")


def parse_mode_from_key_str(key_str: str) -> Optional[Mode]:
    s = (key_str or "").lower()
    if "major" in s:
        return "major"
    if "minor" in s:
        return "minor"
    return None


def simplify_rn_figure(rn: str) -> str:
    """
    Canonicalise RN to a small vocabulary and drop chromatic accidentals:
      - drop anything after '/' (secondary dominants)
      - strip accidentals (#/b) and other decorations
      - keep only root + allowed figured-bass suffix in _ALLOWED_FIGS
    """
    if not rn or rn == "N":
        return "N"

    s = rn.strip().replace("°", "o").replace("ø", "o")
    s = s.split("/")[0]                 # ignore applied dominants for MVP
    s = s.replace("#", "").replace("b", "")  # drop chromatic roots like bVII/#iv

    m = _RN_ROOT_RE.match(s)
    if not m:
        return "N"
    root = m.group(1)

    fig = "".join(_DIGITS_RE.findall(s[len(root):]))

    if fig in ("", "5", "53", "532", "3"):
        fig_out = ""
    elif fig in ("6", "63", "64"):
        fig_out = "63"
    elif "42" in fig:
        fig_out = "42"
    elif "43" in fig:
        fig_out = "43"
    elif "7" in fig or "65" in fig or "75" in fig:
        fig_out = "7"
    else:
        fig_out = ""

    if fig_out not in _ALLOWED_FIGS:
        fig_out = ""

    return f"{root}{fig_out}"


def rn_root(rn_s: str) -> str:
    m = _RN_ROOT_RE.match(rn_s)
    return m.group(1) if m else ""


def rn_to_function(rn_s: str) -> Function:
    """
    Coarse function mapping from simplified RN root.
    """
    if not rn_s or rn_s == "N":
        return "UNK"

    root = rn_root(rn_s)
    if not root:
        return "UNK"

    rl = root.lower()

    # Dominant: V, v, vii°
    if rl == "v" or rl.startswith("vii"):
        return "D"

    # Predominant: ii, iv
    if rl.startswith("ii") or rl == "iv":
        return "PD"

    # Tonic: I/i, iii, vi
    if rl == "i" or rl == "iii" or rl == "vi":
        return "T"

    return "UNK"


def rn_is_diatonic_in_mode(rn_s: str, mode: Mode) -> bool:
    """
    Option A: filter by diatonic root class only (ignore chromatic roots).
    """
    root = rn_root(rn_s)
    if not root:
        return False
    if mode == "major":
        return root in _DIATONIC_MAJOR
    return root in _DIATONIC_MINOR


def majority_vote(items: Sequence[str]) -> str:
    if not items:
        return "N"
    counts: Dict[str, int] = {}
    best = items[0]
    best_c = 0
    for x in items:
        c = counts.get(x, 0) + 1
        counts[x] = c
        if c > best_c:
            best = x
            best_c = c
    return best


def compress_rn_quarters_to_halfbars(rn_quarters: Sequence[str]) -> List[str]:
    """
    Quarter-note RN labels -> half-bar RN labels via majority vote over 2 quarters.
    """
    out: List[str] = []
    n = (len(rn_quarters) // 2) * 2
    for i in range(0, n, 2):
        out.append(majority_vote(rn_quarters[i : i + 2]))
    return out


# ----------------------------
# Generic k-gram model
# ----------------------------

Context = Tuple[str, ...]


@dataclass
class NGramConfig:
    k: int = 3
    alpha: float = 0.25
    seed: Optional[int] = None


@dataclass
class NGramModel:
    k: int
    alpha: float
    vocab: List[str]
    counts_by_order: List[Dict[Context, Dict[str, int]]]

    def _get_counts(self, order: int, ctx: Context) -> Dict[str, int]:
        return self.counts_by_order[order - 1].get(ctx, {})

    def prob(self, token: str, ctx: Context) -> float:
        order = min(self.k, len(ctx) + 1)
        if order > 1:
            ctx = ctx[-(order - 1):]
        else:
            ctx = ()

        counts = self._get_counts(order, ctx)
        if not counts and order > 1:
            return self.prob(token, ctx[1:])

        total = sum(counts.values())
        c = counts.get(token, 0)
        V = len(self.vocab)
        return (c + self.alpha) / (total + self.alpha * V)

    def sample(self, ctx: Context, rng: random.Random) -> str:
        ctx = ctx[-(self.k - 1):] if self.k > 1 else ()
        weights = [self.prob(tok, ctx) for tok in self.vocab]
        s = sum(weights)
        if s <= 0:
            return rng.choice(self.vocab)
        x = rng.random() * s
        acc = 0.0
        for tok, w in zip(self.vocab, weights):
            acc += w
            if x <= acc:
                return tok
        return self.vocab[-1]


def train_ngram(seqs: Iterable[Sequence[str]], cfg: NGramConfig) -> NGramModel:
    vocab_set = set()
    seqs_list: List[List[str]] = []
    for s in seqs:
        s2 = [t for t in s if t]  # defensive
        if not s2:
            continue
        seqs_list.append(s2)
        vocab_set.update(s2)

    vocab = sorted(vocab_set)
    if not vocab:
        raise ValueError("train_ngram: empty vocab (no training sequences after filtering)")

    counts_by_order: List[Dict[Context, Dict[str, int]]] = [dict() for _ in range(cfg.k)]

    def bump(order: int, ctx: Context, tok: str) -> None:
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
# Two-stage harmony model
# ----------------------------

@dataclass
class HarmonyConfig:
    num_bars: int = 8
    k_func: int = 3
    alpha_func: float = 0.25
    seed: Optional[int] = None
    mode: Mode = "major"

    enforce_final_cadence: bool = True
    cadence_pattern: Tuple[Function, Function] = ("D", "T")
    func_vocab: Tuple[Function, ...] = ("T", "PD", "D")


@dataclass
class HarmonyModel:
    cfg: HarmonyConfig
    func_ngram: NGramModel
    rn_by_func: Dict[Function, Dict[str, float]]


def fit_rn_distributions(rn_halfbars: Iterable[Sequence[str]]) -> Dict[Function, Dict[str, float]]:
    counts: Dict[Function, Dict[str, int]] = {"T": {}, "PD": {}, "D": {}}

    for seq in rn_halfbars:
        for rn in seq:
            rn_s = simplify_rn_figure(rn)
            if rn_s == "N":
                continue
            f = rn_to_function(rn_s)
            if f not in counts:
                continue
            d = counts[f]
            d[rn_s] = d.get(rn_s, 0) + 1

    rn_by_func: Dict[Function, Dict[str, float]] = {}
    for f, d in counts.items():
        total = sum(d.values())
        rn_by_func[f] = {} if total == 0 else {k: v / total for k, v in d.items()}
    return rn_by_func


def train_harmony_model(pieces: Iterable[dict], cfg: HarmonyConfig) -> HarmonyModel:
    rn_halfbar_seqs: List[List[str]] = []
    func_seqs: List[List[str]] = []

    for item in pieces:
        item_mode = parse_mode_from_key_str(item.get("key", ""))
        if item_mode != cfg.mode:
            continue

        rn_quarters = item.get("harmony_tokens", [])
        rn_half = compress_rn_quarters_to_halfbars(rn_quarters)

        rn_half_s = [simplify_rn_figure(r) for r in rn_half]
        rn_half_s = [r for r in rn_half_s if r != "N" and rn_is_diatonic_in_mode(r, cfg.mode)]

        func = [rn_to_function(r) for r in rn_half_s]
        func = [f for f in func if f in cfg.func_vocab]

        if len(func) < 4:
            continue

        rn_halfbar_seqs.append(rn_half_s)
        func_seqs.append(func)

    func_ngram = train_ngram(func_seqs, NGramConfig(k=cfg.k_func, alpha=cfg.alpha_func, seed=cfg.seed))
    rn_by_func = fit_rn_distributions(rn_halfbar_seqs)
    return HarmonyModel(cfg=cfg, func_ngram=func_ngram, rn_by_func=rn_by_func)


def _sample_from_categorical(dist: Dict[str, float], rng: random.Random) -> str:
    if not dist:
        return "N"
    items = list(dist.items())
    s = sum(p for _, p in items)
    x = rng.random() * s
    acc = 0.0
    for k, p in items:
        acc += p
        if x <= acc:
            return k
    return items[-1][0]


def sample_function_plan(model: HarmonyModel) -> List[Function]:
    cfg = model.cfg
    rng = random.Random(cfg.seed)

    n_halfbars = cfg.num_bars * 2
    reserve = 2 if cfg.enforce_final_cadence else 0
    target_len = n_halfbars - reserve

    plan: List[Function] = []
    while len(plan) < target_len:
        ctx = tuple(plan[-(model.func_ngram.k - 1):])
        nxt = model.func_ngram.sample(ctx, rng)
        if nxt in cfg.func_vocab:
            plan.append(nxt)

    if cfg.enforce_final_cadence:
        plan.extend(list(cfg.cadence_pattern))
    return plan


def realise_function_plan_to_rn(model: HarmonyModel, func_plan: Sequence[Function]) -> List[str]:
    rng = random.Random(model.cfg.seed)
    rns: List[str] = []
    for f in func_plan:
        dist = model.rn_by_func.get(f, {})
        rns.append(_sample_from_categorical(dist, rng))
    return rns


def sample_harmony_plan(model: HarmonyModel) -> Tuple[List[Function], List[str]]:
    func_plan = sample_function_plan(model)
    rn_plan = realise_function_plan_to_rn(model, func_plan)
    return func_plan, rn_plan