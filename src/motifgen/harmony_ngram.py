
# src/motifgen/harmony_ngram.py
# k-gram counts + smoothing + sampling
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random
import re

# ----------------------------
# Harmony token utilities
# ----------------------------

Function = str  # "T" | "PD" | "D" | "UNK"
_ALLOWED_FIGS = {"", "63", "7", "43", "42"}


def rn_to_function(rn: str) -> Function:
    """
    Map a RomanNumeral figure string to a coarse harmonic function:
      - T  (tonic / tonic-prolongation)
      - PD (predominant)
      - D  (dominant)
      - UNK (unknown / unusable)

    This is intentionally simple and robust for chorale-style RN labels.
    """
    if not rn or rn == "N":
        return "UNK"

    s = rn.strip()

    # Strip common decorations while preserving core root:
    # e.g. "V7", "V65", "vii°6", "V/V" (secondary dominant)
    # We'll decide function from the leading RN token.
    core = s.split("/")[0]  # treat secondary dominants still as dominant-ish
    core = core.replace("ø", "o").replace("°", "o")  # normalize diminished marks

    # Extract the leading roman root (letters i/v)
    # e.g. "V65" -> "V", "ii6" -> "ii", "viio7" -> "viio"
    root = ""
    for ch in core:
        if ch.lower() in ("i", "v"):
            root += ch
        else:
            break

    # If root is empty, give up
    if not root:
        return "UNK"

    root_l = root.lower()

    # Dominant function: V* or vii°*
    if root_l == "v":
        return "PD"
    if root == "V":
        return "D"
    if root_l.startswith("vii"):
        return "D"

    # Predominant: ii or iv/IV
    if root_l.startswith("ii"):
        return "PD"
    if root_l == "iv":
        return "PD"
    if root_l == "iv":  # redundant but explicit
        return "PD"

    # Tonic: i, iii, vi (treat iii as tonic-ish for MVP)
    if root_l == "i":
        return "T"
    if root_l == "iii":
        return "T"
    if root_l == "vi":
        return "T"

    return "UNK"


def simplify_rn_figure(rn: str) -> str:
    """
    Canonicalise a roman numeral figure to a small vocabulary:
      - triads: "" (root), "63" (first inversion)
      - 7ths: "7" (root), "43" (2nd inversion), "42" (3rd inversion)

    Strategy:
      - strip whitespace
      - remove accidentals (#/b) and other non-figure decorations
      - extract the roman root (e.g. I, ii, V, viio)
      - extract any digits (figures) and map to nearest allowed set
    """
    if not rn or rn == "N":
        return "N"

    s = rn.strip()

    # Normalize diminished symbols to a plain marker we keep in root (optional)
    s = s.replace("°", "o").replace("ø", "o")

    # Remove accidentals and non-essential symbols except roman letters + digits + 'o' + '/'
    # We'll separately extract root and digits anyway.
    # (This will drop things like "#42" or "b7" and keep the core)
    # NOTE: leaving '/' in lets us ignore secondary dominants later.
    # But for simplicity, we drop everything after '/'.
    s = s.split("/")[0]

    # Extract roman root: optional 'o' at end for diminished
    m = re.match(r"^([ivIV]+o?)", s)
    if not m:
        return "N"
    root = m.group(1)

    # Extract all digits after root
    digits = re.findall(r"\d+", s[len(root):])
    fig = "".join(digits)  # e.g. "65", "753", "532", "42", "6"

    # Map figures to allowed set
    # Triads:
    if fig in ("", "5", "53", "532", "3"):
        fig_out = ""
    elif fig in ("6", "63", "64"):
        fig_out = "63"  # collapse 6 and 64 to 63
    # 7ths:
    elif "42" in fig:
        fig_out = "42"
    elif "43" in fig:
        fig_out = "43"
    elif "7" in fig or "65" in fig or "75" in fig:
        fig_out = "7"
    else:
        # Unknown -> default to triad root (most stable)
        fig_out = ""

    if fig_out not in _ALLOWED_FIGS:
        fig_out = ""

    return f"{root}{fig_out}"


def majority_vote(items: Sequence[str]) -> str:
    """
    Return the most common item; ties broken by first encountered.
    Empty -> "N".
    """
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
    Convert quarter-note RN labels into half-bar RN labels.

    Assumptions:
      - 4/4: 1 bar = 4 quarters
      - half-bar = 2 quarters (2 beats)

    So we chunk RN labels by 2 and take a majority vote to reduce noise.

    Output length: floor(len(rn_quarters) / 2)
    """
    out: List[str] = []
    n = (len(rn_quarters) // 2) * 2
    for i in range(0, n, 2):
        chunk = rn_quarters[i : i + 2]
        out.append(majority_vote(chunk))
    return out


# ----------------------------
# Generic k-gram model
# ----------------------------

Context = Tuple[str, ...]


@dataclass
class NGramConfig:
    """
    Smoothing + sampling behaviour.
    """
    k: int = 3                 # order (e.g. 2=bigram, 3=trigram)
    alpha: float = 0.25        # add-alpha smoothing
    backoff: bool = True       # if unseen context, back off to shorter context
    seed: Optional[int] = None


@dataclass
class NGramModel:
    """
    A simple categorical n-gram model with add-alpha smoothing and optional backoff.
    Stores counts for all orders 1..k.

    counts_by_order[n][context][token] = count
    where:
      - order 1: context=()
      - order 2: context=(prev,)
      - order 3: context=(prev2, prev1), etc.
    """
    k: int
    alpha: float
    vocab: List[str]
    counts_by_order: List[Dict[Context, Dict[str, int]]]

    def vocab_size(self) -> int:
        return len(self.vocab)

    def _get_counts(self, order: int, ctx: Context) -> Dict[str, int]:
        return self.counts_by_order[order - 1].get(ctx, {})

    def prob(self, token: str, ctx: Context) -> float:
        """
        Smoothed probability P(token | ctx). ctx length must be <= k-1.
        """
        # choose highest order available that matches context length
        order = min(self.k, len(ctx) + 1)
        # optional backoff for unseen contexts
        if self._get_counts(order, ctx) or not self._should_backoff():
            return self._smoothed_prob(order, ctx, token)

        # back off progressively
        if not ctx:
            return self._smoothed_prob(1, (), token)
        return self.prob(token, ctx[1:])

    def _smoothed_prob(self, order: int, ctx: Context, token: str) -> float:
        counts = self._get_counts(order, ctx)
        total = sum(counts.values())
        c = counts.get(token, 0)
        V = self.vocab_size()
        return (c + self.alpha) / (total + self.alpha * V)

    def _should_backoff(self) -> bool:
        # Backoff is enabled if alpha >= 0 and vocab non-empty; caller decides.
        return True

    def sample(self, ctx: Context, rng: random.Random) -> str:
        """
        Sample a token from P(.|ctx) with smoothing+backoff.
        """
        # use highest order feasible with ctx
        ctx = ctx[-(self.k - 1) :] if self.k > 1 else ()
        weights = [self.prob(tok, ctx) for tok in self.vocab]
        # numerical guard
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

    def nll(self, seq: Sequence[str]) -> float:
        """
        Negative log-likelihood (natural log) under the model.
        """
        if not seq:
            return 0.0
        nll = 0.0
        for i, tok in enumerate(seq):
            ctx = tuple(seq[max(0, i - (self.k - 1)) : i])
            p = self.prob(tok, ctx)
            nll -= math.log(max(p, 1e-12))
        return nll

    def perplexity(self, seq: Sequence[str]) -> float:
        """
        Per-token perplexity exp(NLL / N).
        """
        if not seq:
            return float("inf")
        return math.exp(self.nll(seq) / len(seq))


def train_ngram(seqs: Iterable[Sequence[str]], cfg: NGramConfig) -> NGramModel:
    """
    Train n-gram counts for orders 1..k with a fixed vocabulary from the data.
    """
    vocab_set = set()
    for s in seqs:
        vocab_set.update(s)
    vocab = sorted(vocab_set)

    counts_by_order: List[Dict[Context, Dict[str, int]]] = [dict() for _ in range(cfg.k)]

    def bump(order: int, ctx: Context, tok: str) -> None:
        d = counts_by_order[order - 1].setdefault(ctx, {})
        d[tok] = d.get(tok, 0) + 1

    for seq in seqs:
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
    """
    Configuration for function-level harmony planning + chord realisation.

    We model function sequence at half-bar granularity (2 chords per bar).
    """
    num_bars: int = 8
    k_func: int = 3
    alpha_func: float = 0.25
    seed: Optional[int] = None

    # Cadence constraints
    enforce_final_cadence: bool = True
    # default cadence at half-bar resolution: last two half-bars are D -> T
    cadence_pattern: Tuple[Function, Function] = ("D", "T")

    # Supported function vocab
    func_vocab: Tuple[Function, ...] = ("T", "PD", "D")


@dataclass
class HarmonyModel:
    """
    - func_ngram: N-gram over function tokens (T/PD/D) at half-bar resolution
    - rn_by_func: empirical categorical distribution of RN figures conditioned on function
    """
    cfg: HarmonyConfig
    func_ngram: NGramModel
    rn_by_func: Dict[Function, Dict[str, float]]  # func -> {rn: prob}


def fit_rn_distributions(
    rn_halfbars: Iterable[Sequence[str]],
) -> Dict[Function, Dict[str, float]]:
    """
    Build P(RN | Function) from half-bar RN sequences using rn_to_function mapping.
    """
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
            d[rn] = d.get(rn, 0) + 1

    # Normalize with tiny smoothing to avoid zeros
    rn_by_func: Dict[Function, Dict[str, float]] = {}
    for f, d in counts.items():
        total = sum(d.values())
        if total == 0:
            rn_by_func[f] = {}
            continue
        rn_by_func[f] = {rn: c / total for rn, c in d.items()}

    return rn_by_func


def train_harmony_model(
    pieces: Iterable[dict],
    cfg: HarmonyConfig,
) -> HarmonyModel:
    """
    Train a harmony model from processed dataset items produced by dataset.py.

    Each item is expected to have:
      - item["harmony_tokens"]: quarter-note RN labels (strings)
    """
    rn_halfbar_seqs: List[List[str]] = []
    func_seqs: List[List[str]] = []

    for item in pieces:
        rn_quarters = item.get("harmony_tokens", [])
        rn_half = compress_rn_quarters_to_halfbars(rn_quarters)

        rn_half = [simplify_rn_figure(r) for r in rn_half]
        rn_half = [r for r in rn_half if r != "N"]

        # drop UNK tokens for function sequence training by filtering
        func = [rn_to_function(r) for r in rn_half]
        func = [f for f in func if f in cfg.func_vocab]
        if len(func) < 4:
            continue
        rn_halfbar_seqs.append(rn_half)
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
    """
    Sample a function plan at half-bar resolution: length = 2 * num_bars.
    Applies a simple final cadence constraint if enabled.
    """
    cfg = model.cfg
    rng = random.Random(cfg.seed)

    n_halfbars = cfg.num_bars * 2
    plan: List[Function] = []

    # If enforcing cadence, reserve last two half-bars.
    reserve = 2 if cfg.enforce_final_cadence else 0
    target_len = n_halfbars - reserve

    while len(plan) < target_len:
        ctx = tuple(plan[-(model.func_ngram.k - 1) :])
        nxt = model.func_ngram.sample(ctx, rng)
        # safety: only allow known vocab
        if nxt not in cfg.func_vocab:
            continue
        plan.append(nxt)

    if cfg.enforce_final_cadence:
        plan.extend(list(cfg.cadence_pattern))  # ["D","T"]

    return plan


def realise_function_plan_to_rn(model: HarmonyModel, func_plan: Sequence[Function]) -> List[str]:
    """
    Realise each function token into a specific RN figure by sampling P(RN|Function).
    """
    rng = random.Random(model.cfg.seed)
    rns: List[str] = []
    for f in func_plan:
        dist = model.rn_by_func.get(f, {})
        rns.append(_sample_from_categorical(dist, rng))
    return rns


# ----------------------------
# Convenience: end-to-end harmony sampling
# ----------------------------

def sample_harmony_plan(model: HarmonyModel) -> Tuple[List[Function], List[str]]:
    """
    Returns:
      - function plan (T/PD/D) at half-bar resolution
      - realised RN plan at half-bar resolution
    """
    func_plan = sample_function_plan(model)
    rn_plan = realise_function_plan_to_rn(model, func_plan)
    return func_plan, rn_plan

if __name__ == "__main__":
    # Example usage: train on dataset and sample a harmony plan
    pieces = [{"id": "bach/bwv8.6", "key": "B major", "melody_tokens": ["BAR", "R:1.0", "N:0:0.5", "N:0:0.5", "N:5:1.0", "N:0:1.0", "BAR", "N:2:1.0", "N:0:1.0", "N:10:2.0", "N:0:0.5", "N:10:0.5", "BAR", "N:9:1.0", "N:7:1.0", "R:1.0", "N:0:0.5", "N:10:0.5", "BAR", "N:9:1.0", "N:2:0.5", "N:0:0.5", "N:11:0.5", "N:7:0.5", "N:0:1.0", "BAR", "N:0:1.0", "N:11:1.0", "N:0:2.0", "BAR", "N:0:1.0", "N:11:1.0", "N:0:2.0", "BAR", "R:1.0", "N:7:0.5", "N:9:0.5", "N:10:1.0", "N:9:1.0", "BAR", "N:2:2.0", "N:4:0.5", "N:1:2.0", "BAR", "R:1.0", "N:2:0.5", "N:0:0.5", "N:11:1.0", "N:0:1.0", "BAR", "N:0:1.0", "N:11:1.0", "N:0:1.0", "N:7:0.5", "N:7:0.5", "BAR", "N:0:2.0", "N:10:0.5", "N:9:1.0", "N:2:1.0", "BAR", "N:1:1.0", "N:2:2.0", "N:1:1.0", "BAR", "N:2:2.0", "R:1.0", "N:5:0.5", "N:0:0.5", "BAR", "N:2:1.0", "N:9:0.5", "N:10:0.5", "N:0:2.0", "N:10:0.5", "BAR", "N:9:1.0", "N:7:0.5", "N:5:0.5", "N:4:1.0", "N:5:1.0", "BAR", "N:5:1.0", "N:4:1.0", "N:5:2.0"], "harmony_tokens": ["iv", "iv", "i", "iv", "IV", "bVII", "IV", "v75b3", "I6b5", "IV", "I742", "iii", "iiiob5", "IV", "IV7", "iii7", "I", "i54", "V", "I", "I", "ii652", "V", "I", "I532", "iii", "vi6", "vb3", "vi43", "iiiø7b53", "iiio6", "VI", "VI6#42", "iv", "IV7", "V42", "I752", "i4", "V", "I", "v4", "I6", "Ib753", "vi43", "ii42", "VI", "#vø7", "ii54", "VI", "ii", "#i7b2", "ii", "IV6b5", "ii42", "bVII6", "IV6", "I6b5", "IV", "bvii", "I42", "viiob753", "iv4", "I", "IV"]},
                {"id": "bach/bwv67.7", "key": "A major", "melody_tokens": ["BAR", "N:4:1.0", "N:0:1.0", "N:2:1.0", "N:4:1.0", "BAR", "N:7:1.0", "N:5:1.0", "N:5:1.0", "N:4:1.0", "BAR", "N:7:1.0", "N:5:1.0", "N:4:1.0", "N:2:1.0", "BAR", "N:2:1.0", "N:4:2.0", "BAR", "N:2:1.0", "N:2:1.0", "N:2:1.0", "N:4:1.0", "BAR", "N:2:1.0", "N:0:1.0", "N:2:1.0", "N:11:1.0", "BAR", "N:11:1.0", "N:0:1.0", "N:2:1.0", "N:4:1.0", "BAR", "N:2:0.5", "N:4:0.5", "N:5:1.0", "N:4:1.0", "N:2:1.0", "BAR", "N:0:2.0", "R:1.0"], "harmony_tokens": ["I", "I", "IV42", "viio6", "I", "V6", "IV6", "V65", "I", "I6", "IV752", "I", "ii65", "V", "I", "I", "I", "V742", "V6", "V", "I", "V7", "vi", "viio6", "III", "III", "vi", "V", "I", "iii64", "IV6", "I64", "ii65", "V7", "I", "I"]}]
    cfg = HarmonyConfig(num_bars=8, k_func=3, alpha_func=0.25, seed=42)
    model = train_harmony_model(pieces, cfg)

    func_plan, rn_plan = sample_harmony_plan(model)
    print("Sampled function plan:", func_plan)
    print("Realised RN plan:", rn_plan)