import argparse
import csv
import json
from pathlib import Path

from summarize_experiments import EXPERIMENT_LABELS, repo_root


FIELDNAMES = [
    "dataset",
    "group",
    "experiment_id",
    "experiment_label",
    "baseline_root",
    "candidate_root",
    "baseline_f1",
    "candidate_f1",
    "delta_f1",
    "baseline_recall",
    "candidate_recall",
    "delta_recall",
    "baseline_hit1",
    "candidate_hit1",
    "delta_hit1",
    "baseline_hit3",
    "candidate_hit3",
    "delta_hit3",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare experiment summaries from two branch-specific output roots.")
    parser.add_argument("--baseline_root", required=True, help="Reference output root, e.g. output_branches/main")
    parser.add_argument("--candidate_root", required=True, help="Candidate output root, e.g. output_branches/feat-sparse-gat")
    parser.add_argument("--dataset", default="SMD", help="Dataset name. Default: SMD")
    parser.add_argument("--summary_name", default="experiment_summary.csv", help="Summary CSV filename inside each report directory")
    parser.add_argument("--baseline_report_dir", default=None, help="Directory containing baseline summary CSV. Defaults to reports_branch/<root-name>")
    parser.add_argument("--candidate_report_dir", default=None, help="Directory containing candidate summary CSV. Defaults to reports_branch/<root-name>")
    parser.add_argument("--output_dir", default="reports_compare", help="Directory to write comparison CSV/Markdown into.")
    return parser.parse_args()


def safe_root_name(output_root):
    return Path(output_root).name


def resolve_report_dir(explicit_dir, output_root):
    if explicit_dir is not None:
        return repo_root() / explicit_dir
    return repo_root() / "reports_branch" / safe_root_name(output_root)


def load_rows(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def key_for_row(row):
    return row["dataset"], row["group"], row["experiment_id"]


def parse_float(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def format_value(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def make_comparison_row(dataset, group, experiment_id, baseline_row, candidate_row, baseline_root, candidate_root):
    baseline_f1 = parse_float(baseline_row, "f1")
    candidate_f1 = parse_float(candidate_row, "f1")
    baseline_recall = parse_float(baseline_row, "recall")
    candidate_recall = parse_float(candidate_row, "recall")
    baseline_hit1 = parse_float(baseline_row, "hit1")
    candidate_hit1 = parse_float(candidate_row, "hit1")
    baseline_hit3 = parse_float(baseline_row, "hit3")
    candidate_hit3 = parse_float(candidate_row, "hit3")

    return {
        "dataset": dataset,
        "group": group,
        "experiment_id": experiment_id,
        "experiment_label": EXPERIMENT_LABELS.get(experiment_id, experiment_id),
        "baseline_root": baseline_root,
        "candidate_root": candidate_root,
        "baseline_f1": baseline_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": None if baseline_f1 is None or candidate_f1 is None else candidate_f1 - baseline_f1,
        "baseline_recall": baseline_recall,
        "candidate_recall": candidate_recall,
        "delta_recall": None if baseline_recall is None or candidate_recall is None else candidate_recall - baseline_recall,
        "baseline_hit1": baseline_hit1,
        "candidate_hit1": candidate_hit1,
        "delta_hit1": None if baseline_hit1 is None or candidate_hit1 is None else candidate_hit1 - baseline_hit1,
        "baseline_hit3": baseline_hit3,
        "candidate_hit3": candidate_hit3,
        "delta_hit3": None if baseline_hit3 is None or candidate_hit3 is None else candidate_hit3 - baseline_hit3,
    }


def write_csv(rows, path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_markdown(rows):
    if not rows:
        return "No overlapping experiment rows were found between the two summaries.\n"

    lines = [
        "| Experiment | Group | Base F1 | Cand F1 | dF1 | Base Recall | Cand Recall | dRecall | Base Hit@1 | Cand Hit@1 | dHit@1 | Base Hit@3 | Cand Hit@3 | dHit@3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {exp} | {group} | {bf1} | {cf1} | {df1} | {br} | {cr} | {dr} | {bh1} | {ch1} | {dh1} | {bh3} | {ch3} | {dh3} |".format(
                exp=row["experiment_id"],
                group=row["group"],
                bf1=format_value(row["baseline_f1"]),
                cf1=format_value(row["candidate_f1"]),
                df1=format_value(row["delta_f1"]),
                br=format_value(row["baseline_recall"]),
                cr=format_value(row["candidate_recall"]),
                dr=format_value(row["delta_recall"]),
                bh1=format_value(row["baseline_hit1"]),
                ch1=format_value(row["candidate_hit1"]),
                dh1=format_value(row["delta_hit1"]),
                bh3=format_value(row["baseline_hit3"]),
                ch3=format_value(row["candidate_hit3"]),
                dh3=format_value(row["delta_hit3"]),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    baseline_report_dir = resolve_report_dir(args.baseline_report_dir, args.baseline_root)
    candidate_report_dir = resolve_report_dir(args.candidate_report_dir, args.candidate_root)
    baseline_csv = baseline_report_dir / args.summary_name
    candidate_csv = candidate_report_dir / args.summary_name

    baseline_rows = {key_for_row(row): row for row in load_rows(baseline_csv)}
    candidate_rows = {key_for_row(row): row for row in load_rows(candidate_csv)}
    shared_keys = sorted(set(baseline_rows) & set(candidate_rows))

    comparison_rows = []
    for dataset, group, experiment_id in shared_keys:
        if dataset != args.dataset:
            continue
        comparison_rows.append(
            make_comparison_row(
                dataset,
                group,
                experiment_id,
                baseline_rows[(dataset, group, experiment_id)],
                candidate_rows[(dataset, group, experiment_id)],
                args.baseline_root,
                args.candidate_root,
            )
        )

    output_dir = repo_root() / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_name = safe_root_name(args.baseline_root)
    candidate_name = safe_root_name(args.candidate_root)
    csv_path = output_dir / f"compare_{baseline_name}_vs_{candidate_name}.csv"
    md_path = output_dir / f"compare_{baseline_name}_vs_{candidate_name}.md"
    meta_path = output_dir / f"compare_{baseline_name}_vs_{candidate_name}.json"

    write_csv(comparison_rows, csv_path)
    md_path.write_text(build_markdown(comparison_rows), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "baseline_root": args.baseline_root,
                "candidate_root": args.candidate_root,
                "baseline_report_dir": str(baseline_report_dir),
                "candidate_report_dir": str(candidate_report_dir),
                "rows": len(comparison_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(comparison_rows)} rows to {csv_path}")
    print(f"Wrote markdown table to {md_path}")


if __name__ == "__main__":
    main()
