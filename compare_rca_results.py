import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from analyze_results import compute_rca_hitk, load_latest_summary, load_summary_metrics


def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare RCA metrics across MTAD-GAT run directories.")
    parser.add_argument("--dataset", type=str.upper, required=True, help="Dataset name, e.g. SMD")
    parser.add_argument(
        "--groups",
        type=parse_list_arg,
        default=["1-3"],
        help="Comma-separated groups. For SMD, examples: 1-2,1-3,2-1,3-6",
    )
    parser.add_argument(
        "--run_dirs",
        type=parse_list_arg,
        default=None,
        help="Optional explicit run directories to compare. When omitted, latest runs are discovered per group.",
    )
    parser.add_argument(
        "--latest_per_group",
        type=int,
        default=3,
        help="How many latest run directories to compare per group when --run_dirs is not provided.",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="output",
        help="Root directory containing training outputs.",
    )
    parser.add_argument(
        "--comment_filter",
        type=str,
        default=None,
        help="Optional substring filter applied to config.txt comment when auto-discovering runs.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="rca_comparison_output",
        help="Where to save comparison CSV/Markdown files.",
    )
    return parser


def infer_model_name(config: Dict) -> str:
    if config.get("use_adaptive_sparse_feat_gat", False):
        recon_model = str(config.get("recon_model", "gru")).lower()
        return f"mtad_gat_adaptive_sparse_{recon_model}"

    recon_model = str(config.get("recon_model", "gru")).lower()
    use_sparsemax = bool(config.get("use_sparsemax", False))
    use_node_embedding = bool(config.get("use_node_embedding", False))

    parts = ["mtad_gat"]
    if use_sparsemax:
        parts.append("sparsemax")
    parts.append(recon_model)
    if use_node_embedding:
        parts.append("node")
    return "_".join(parts)


def load_config(run_dir: Path) -> Dict:
    config_path = run_dir / "config.txt"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_run_dirs(dataset: str, groups: List[str], runs_root: Path, latest_per_group: int, comment_filter: str) -> List[Path]:
    discovered: List[Path] = []
    for group in groups:
        group_dir = runs_root / dataset / group
        if not group_dir.is_dir():
            raise FileNotFoundError(f"Group output directory not found: {group_dir}")

        run_dirs = [path for path in group_dir.iterdir() if path.is_dir() and path.name != "logs"]
        run_dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)

        selected: List[Path] = []
        for run_dir in run_dirs:
            try:
                config = load_config(run_dir)
            except FileNotFoundError:
                continue
            comment = str(config.get("comment", ""))
            if comment_filter and comment_filter not in comment:
                continue
            selected.append(run_dir)
            if len(selected) >= latest_per_group:
                break

        if len(selected) == 0:
            raise FileNotFoundError(f"No matching runs found under {group_dir}")
        discovered.extend(selected)
    return discovered


def collect_rows(dataset: str, run_dirs: List[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        config = load_config(run_dir)
        group = config.get("group")
        model = infer_model_name(config)
        summary = load_latest_summary(str(run_dir)) or {}
        epsilon_f1, precision, recall = load_summary_metrics(str(run_dir))
        bf_f1 = summary.get("bf_result", {}).get("f1")
        pot_f1 = summary.get("pot_result", {}).get("f1")
        hit1, hit3, hit5 = compute_rca_hitk(str(run_dir), dataset, group)
        rows.append(
            {
                "dataset": dataset,
                "group": group,
                "model": model,
                "run_id": run_dir.name,
                "comment": config.get("comment", ""),
                "epsilon_f1": epsilon_f1,
                "bf_f1": bf_f1,
                "pot_f1": pot_f1,
                "precision": precision,
                "recall": recall,
                "hit1": hit1,
                "hit3": hit3,
                "hit5": hit5,
                "run_dir": str(run_dir),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [headers, ["---"] * len(headers)]
    for _, row in df.iterrows():
        rendered = []
        for col in headers:
            value = row[col]
            if pd.isna(value):
                rendered.append("N/A")
            elif isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main():
    parser = build_parser()
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if args.run_dirs is not None:
        run_dirs = [Path(item) for item in args.run_dirs]
    else:
        run_dirs = discover_run_dirs(
            dataset=args.dataset,
            groups=args.groups,
            runs_root=runs_root,
            latest_per_group=args.latest_per_group,
            comment_filter=args.comment_filter,
        )

    per_run_df = collect_rows(args.dataset, run_dirs)
    if per_run_df.empty:
        raise RuntimeError("No RCA rows were collected.")

    summary_df = (
        per_run_df.groupby("model", as_index=False)[
            ["epsilon_f1", "bf_f1", "pot_f1", "precision", "recall", "hit1", "hit3", "hit5"]
        ]
        .mean()
        .sort_values(["hit1", "hit3", "hit5", "epsilon_f1", "bf_f1"], ascending=False)
        .reset_index(drop=True)
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run_df.to_csv(output_dir / "per_run_rca.csv", index=False)
    summary_df.to_csv(output_dir / "model_rca_summary.csv", index=False)

    markdown_lines = [
        "# RCA comparison",
        "",
        "## Aggregated by model",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Per run details",
        "",
        dataframe_to_markdown(per_run_df),
        "",
    ]
    (output_dir / "rca_comparison.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    print(f"Saved RCA comparison to {output_dir}")
    print("\nAggregated by model:")
    print(dataframe_to_markdown(summary_df))


if __name__ == "__main__":
    main()
