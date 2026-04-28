import argparse
import csv
import json
import os
from pathlib import Path

from analyze_results import compute_rca_hitk


DEFAULT_GROUPS = ["1-2", "1-3", "2-1", "3-6"]
DEFAULT_EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5", "E6"]
EXPERIMENT_LABELS = {
    "E1": "Dense + GRU + GATv2",
    "E2": "Sparse + NodeEmbed + VAE + GATv2",
    "E3": "Sparse + NodeEmbed + GRU + GATv2",
    "E4": "Sparse + NodeEmbed + VAE + GAT",
    "E5": "Dense + VAE + GATv2",
    "E6": "Sparse + no NodeEmbed + VAE + GATv2",
}
FIELDNAMES = [
    "output_root",
    "dataset",
    "group",
    "experiment_id",
    "experiment_label",
    "run_dir",
    "timestamp",
    "metric_block",
    "f1",
    "precision",
    "recall",
    "latency",
    "threshold",
    "TP",
    "FP",
    "FN",
    "use_adaptive_sparse_feat_gat",
    "use_node_embedding",
    "recon_model",
    "use_gatv2",
    "comment",
    "hit1",
    "hit3",
    "hit5",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize MTAD-GAT experiment outputs.")
    parser.add_argument("--dataset", default="SMD", help="Dataset name. Default: SMD")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=DEFAULT_GROUPS,
        help="Groups to summarize. Default: 1-2 1-3 2-1 3-6",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment IDs to summarize. Default: E1 E2 E3 E4 E5 E6",
    )
    parser.add_argument(
        "--metric",
        default="epsilon_result",
        choices=["epsilon_result", "pot_result", "bf_result"],
        help="Which detection metric block to read from summary.txt",
    )
    parser.add_argument(
        "--output_root",
        default="output",
        help="Root directory that stores experiment outputs.",
    )
    parser.add_argument(
        "--output_dir",
        default="reports",
        help="Directory to write CSV/Markdown summaries into.",
    )
    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parent


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_run_dirs(base_dir):
    if not base_dir.exists():
        return []
    dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit() is False]
    timestamped = [p for p in dirs if len(p.name) == 15 and p.name[8] == "_"]
    return sorted(timestamped, key=lambda p: p.stat().st_mtime, reverse=True)


def get_latest_run_for_experiment(output_root, dataset, group, experiment_id):
    base_dir = repo_root() / output_root / dataset / group
    for run_dir in iter_run_dirs(base_dir):
        config_path = run_dir / "config.txt"
        if not config_path.exists():
            continue
        try:
            config = load_json(config_path)
        except Exception:
            continue
        if config.get("comment") == experiment_id:
            return run_dir, config
    return None, None


def safe_metric(metric_dict, key):
    value = metric_dict.get(key)
    if value is None:
        return ""
    return value


def format_value(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_row(output_root, dataset, group, experiment_id, run_dir, config, metric_key):
    summary_path = run_dir / "summary.txt"
    if not summary_path.exists():
        return None

    summary = load_json(summary_path)
    metric_dict = summary.get(metric_key, {})
    if not metric_dict:
        return None

    hit1, hit3, hit5 = compute_rca_hitk(str(run_dir), dataset, group)

    return {
        "output_root": output_root,
        "dataset": dataset,
        "group": group,
        "experiment_id": experiment_id,
        "experiment_label": EXPERIMENT_LABELS.get(experiment_id, experiment_id),
        "run_dir": str(run_dir),
        "timestamp": run_dir.name,
        "metric_block": metric_key,
        "f1": safe_metric(metric_dict, "f1"),
        "precision": safe_metric(metric_dict, "precision"),
        "recall": safe_metric(metric_dict, "recall"),
        "latency": safe_metric(metric_dict, "latency"),
        "threshold": safe_metric(metric_dict, "threshold"),
        "TP": safe_metric(metric_dict, "TP"),
        "FP": safe_metric(metric_dict, "FP"),
        "FN": safe_metric(metric_dict, "FN"),
        "use_adaptive_sparse_feat_gat": config.get("use_adaptive_sparse_feat_gat", ""),
        "use_node_embedding": config.get("use_node_embedding", ""),
        "recon_model": config.get("recon_model", ""),
        "use_gatv2": config.get("use_gatv2", ""),
        "comment": config.get("comment", ""),
        "hit1": "" if hit1 is None else hit1,
        "hit3": "" if hit3 is None else hit3,
        "hit5": "" if hit5 is None else hit5,
    }


def write_csv(rows, path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_markdown(rows):
    if not rows:
        return "No matching experiment runs were found.\n"
    header = [
        "| Output Root | Experiment | Group | F1 | Precision | Recall | Latency | Hit@1 | Hit@3 | Hit@5 | Run |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    body = []
    for row in rows:
        body.append(
            "| `{root}` | {exp} | {group} | {f1} | {precision} | {recall} | {latency} | {hit1} | {hit3} | {hit5} | `{run}` |".format(
                root=row["output_root"],
                exp=row["experiment_id"],
                group=row["group"],
                f1=format_value(row["f1"]),
                precision=format_value(row["precision"]),
                recall=format_value(row["recall"]),
                latency=format_value(row["latency"]),
                hit1=format_value(row["hit1"]),
                hit3=format_value(row["hit3"]),
                hit5=format_value(row["hit5"]),
                run=row["timestamp"],
            )
        )
    return "\n".join(header + body) + "\n"


def main():
    args = parse_args()
    rows = []

    for experiment_id in args.experiments:
        for group in args.groups:
            run_dir, config = get_latest_run_for_experiment(args.output_root, args.dataset, group, experiment_id)
            if run_dir is None:
                continue
            row = build_row(args.output_root, args.dataset, group, experiment_id, run_dir, config, args.metric)
            if row is not None:
                rows.append(row)

    rows.sort(key=lambda row: (row["experiment_id"], row["group"]))

    output_dir = repo_root() / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "experiment_summary.csv"
    md_path = output_dir / "experiment_summary.md"

    write_csv(rows, csv_path)
    md_path.write_text(build_markdown(rows), encoding="utf-8")

    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote markdown table to {md_path}")


if __name__ == "__main__":
    main()
