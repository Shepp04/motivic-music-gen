from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Keep matplotlib/font cache inside the repo so batch runs do not warn about unwritable home dirs.
_MPL_CACHE_DIR = Path(".matplotlib-cache")
_MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR.resolve()))

from src.motifgen import eval as eval_mod
from src.motifgen.generate import (
    GenerateConfig,
    default_system_configs,
    generate_many_systems,
    load_jsonl,
    piece_to_jsonable,
    train_models,
)


SYSTEM_LABELS = {
    "baseline_melody_only": "Baseline",
    "ablation_structure_only": "Ablation",
    "full": "Full",
}

# Smaller table intended for direct use in the report.
REPORT_METRICS = [
    ("ssm_score", "SSM"),
    ("sb_chord_tone_rate", "SBCTR"),
    ("diff_mean_abs_interval", "DeltaMeanInt"),
    ("motif_coverage", "MotifCov"),
    ("motif_dispersion_var", "MotifDispVar"),
    ("cadence_pass", "Cadence"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline + ablation + full-system outputs and evaluate them together."
    )
    parser.add_argument("--train-jsonl", default="data/train.jsonl")
    parser.add_argument("--val-jsonl", default="data/val.jsonl")
    parser.add_argument("--test-jsonl", default="data/test.jsonl")
    parser.add_argument("--out-dir", default="outputs/batch_eval")

    parser.add_argument("--num-pieces", type=int, default=30, help="Pieces to generate per system.")
    parser.add_argument("--seed-start", type=int, default=1000, help="First generation seed.")
    parser.add_argument("--train-seed", type=int, default=123, help="Seed used for model training.")

    parser.add_argument("--mode", choices=("major", "minor"), default="major")
    parser.add_argument("--num-bars", type=int, default=8)
    parser.add_argument("--density", type=float, default=0.50)
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Also render interval/metric plots and one SSM heatmap per system.",
    )
    return parser.parse_args()


def save_jsonl(items: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def flatten_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flat_cols: List[str] = []
    for col in out.columns:
        if not isinstance(col, tuple):
            flat_cols.append(str(col))
            continue
        head, tail = col
        if tail in ("", None):
            flat_cols.append(str(head))
        else:
            flat_cols.append(f"{head}_{tail}")
    out.columns = flat_cols
    return out


def format_mean_std(series: pd.Series) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return "N/A"

    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return f"{mean:.3f} ± {std:.3f}"


def build_report_table(df_per_piece: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    grouped = df_per_piece.groupby(["system", "mode"], dropna=False)

    for (system, mode), group in grouped:
        row = {
            "System": SYSTEM_LABELS.get(str(system), str(system)),
            "Mode": mode,
            "N": int(len(group)),
        }
        for metric, label in REPORT_METRICS:
            row[label] = format_mean_std(group[metric])
        rows.append(row)

    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_latex_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    cols = list(df.columns)
    align = "l" + "c" * (len(cols) - 1)

    def latex_cell(value: object) -> str:
        s = str(value)
        if s == "N/A":
            return r"\textemdash"
        return s.replace(" ± ", r" $\pm$ ")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Batch evaluation summary across baseline, ablation, and full systems.}",
        r"\label{tab:batch-eval-summary}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(cols) + r" \\",
        r"\hline",
    ]

    for row in df.itertuples(index=False, name=None):
        lines.append(" & ".join(latex_cell(x) for x in row) + r" \\")

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.num_pieces))
    systems = default_system_configs()

    gcfg = GenerateConfig(
        num_bars=args.num_bars,
        units_per_beat=4,
        beats_per_bar=4,
        density=args.density,
        mode=args.mode,
        motif_dur_units=(16,),
        infill_dur_set=(2.0, 1.0, 0.5),
    )
    eval_cfg = eval_mod.EvalConfig(
        units_per_beat=gcfg.units_per_beat,
        beats_per_bar=gcfg.beats_per_bar,
    )

    train_items = load_jsonl(args.train_jsonl)
    val_items = load_jsonl(args.val_jsonl)
    test_items = load_jsonl(args.test_jsonl)
    corpus = eval_mod.compute_corpus_stats(val_items + test_items, cfg=eval_cfg)

    models = train_models(train_items, cfg=gcfg, seed=args.train_seed)
    pieces = generate_many_systems(models=models, gcfg=gcfg, systems=systems, seeds=seeds)

    df_per_piece, df_summary = eval_mod.evaluate_many(pieces, cfg=eval_cfg, corpus=corpus)
    df_summary_flat = flatten_summary_columns(df_summary)
    report_table = build_report_table(df_per_piece)

    generated_jsonl = save_jsonl(
        (piece_to_jsonable(p) for p in pieces),
        out_dir / "generated_all_systems.jsonl",
    )
    save_json(
        {
            "num_pieces_per_system": args.num_pieces,
            "seed_start": args.seed_start,
            "train_seed": args.train_seed,
            "mode": args.mode,
            "num_bars": args.num_bars,
            "density": args.density,
            "systems": [s.name for s in systems],
            "seeds": seeds,
        },
        out_dir / "run_config.json",
    )

    df_per_piece.to_csv(out_dir / "per_piece_metrics.csv", index=False)
    df_summary_flat.to_csv(out_dir / "summary_flat.csv", index=False)
    report_table.to_csv(out_dir / "report_table.csv", index=False)
    write_markdown_table(report_table, out_dir / "report_table.md")
    write_latex_table(report_table, out_dir / "report_table.tex")

    if args.make_plots:
        eval_mod.plot_interval_hist(pieces=pieces, cfg=eval_cfg, out_dir=out_dir)

        for metric, filename, title in [
            ("ssm_score", "ssm_score.png", "SSM structure score"),
            ("motif_coverage", "motif_coverage.png", "Motif coverage"),
            ("cadence_pass", "cadence_pass.png", "Cadence pass rate"),
            ("diff_mean_abs_interval", "diff_mean_abs_interval.png", "Distance to corpus mean interval"),
        ]:
            if metric in df_per_piece.columns and df_per_piece[metric].notna().any():
                eval_mod.plot_metric_bars(
                    df_per_piece=df_per_piece,
                    metric=metric,
                    out_path=out_dir / filename,
                    title=title,
                )

        for system_name in df_per_piece["system"].dropna().unique():
            piece = next((p for p in pieces if p.get("system") == system_name), None)
            if piece is None:
                continue
            _metrics, ssm = eval_mod.evaluate_piece(piece, cfg=eval_cfg, corpus=corpus)
            if ssm is not None:
                eval_mod.plot_ssm(
                    ssm=ssm,
                    out_path=out_dir / f"ssm_example_{system_name}.png",
                    title=f"SSM example ({system_name})",
                )

    print(f"Wrote combined JSONL to: {generated_jsonl}")
    print(f"Wrote raw metrics to: {out_dir / 'per_piece_metrics.csv'}")
    print(f"Wrote flattened summary to: {out_dir / 'summary_flat.csv'}")
    print(f"Wrote report table to: {out_dir / 'report_table.csv'}")
    print(f"Wrote LaTeX table to: {out_dir / 'report_table.tex'}")


if __name__ == "__main__":
    main()
