import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MTAD-GAT latest run outputs.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. SMD, WADI, MSL")
    parser.add_argument("--group", type=str, default=None, help="Optional group, e.g. 1-1")
    parser.add_argument(
        "--output_root",
        type=str,
        default="output",
        help="Root directory that stores experiment outputs.",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default=None,
        help="Optional explicit interpretation label file path for RCA evaluation.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Optional explicit run directory. If not provided, latest run is used.",
    )
    parser.add_argument(
        "--threshold_method",
        type=str,
        default="pot_result",
        choices=["epsilon_result", "pot_result", "bf_result"],
        help="Which threshold result to report in summary_report.txt.",
    )
    parser.add_argument(
        "--rca_eval_unit",
        type=str,
        default="timestamp",
        choices=["timestamp", "window"],
        help="Interpretation-label alignment unit for RCA evaluation.",
    )
    return parser.parse_args()


def get_base_dir(output_root, dataset, group):
    if group:
        return os.path.join(output_root, dataset, group)
    return os.path.join(output_root, dataset)


def get_repo_root():
    return os.path.dirname(os.path.abspath(__file__))


def get_latest_target_dir(base_dir):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError("Base output directory not found: {}".format(base_dir))

    subdirs = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path):
            subdirs.append(full_path)

    if not subdirs:
        raise FileNotFoundError("No run directories found under {}".format(base_dir))

    subdirs.sort(key=os.path.getmtime, reverse=True)
    return subdirs[0]


def load_summary_metrics(target_dir, threshold_method="pot_result"):
    summary_files = []
    for name in os.listdir(target_dir):
        if name.startswith("summary") and name.endswith(".txt"):
            summary_files.append(os.path.join(target_dir, name))

    if not summary_files:
        warnings.warn("No summary*.txt found under {}".format(target_dir))
        return None, None, None

    summary_files.sort(key=os.path.getmtime, reverse=True)
    for summary_path in summary_files:
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            result = summary.get(threshold_method, {})
            return result.get("f1"), result.get("precision"), result.get("recall")
        except Exception:
            continue

    warnings.warn("No valid JSON summary file found under {}".format(target_dir))
    return None, None, None


def infer_group_from_target_dir(dataset, group, target_dir):
    if group is not None or dataset.upper() != "SMD" or target_dir is None:
        return group

    normalized = os.path.normpath(target_dir)
    parts = normalized.split(os.sep)
    for i, part in enumerate(parts[:-1]):
        if part.upper() == "SMD" and i + 1 < len(parts):
            candidate = parts[i + 1]
            if re.match(r"^\d+-\d+$", candidate):
                return candidate
    return group


def resolve_label_path(dataset, group, label_path_override=None):
    if label_path_override is not None:
        if os.path.isfile(label_path_override):
            return label_path_override
        warnings.warn("Provided label path does not exist: {}".format(label_path_override))
        return None

    ds_upper = dataset.upper()
    repo_root = get_repo_root()
    if ds_upper == "SMD":
        if group is None:
            warnings.warn("SMD requires --group to locate interpretation labels.")
            return None

        candidate_paths = [
            os.path.join(repo_root, "datasets", "ServerMachineDataset", "interpretation_label", "machine-{}.txt".format(group)),
            os.path.join(repo_root, "datasets", "SMD", "interpretation_label", "machine-{}.txt".format(group)),
            os.path.join("/root/mtad/datasets/SMD/interpretation_label", "machine-{}.txt".format(group)),
            os.path.join("/root/mtad/datasets/ServerMachineDataset/interpretation_label", "machine-{}.txt".format(group)),
        ]
        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                return candidate
        return None

    candidate_paths = [
        os.path.join(repo_root, "datasets", "data", "{}_interpretation_label.txt".format(dataset.lower())),
        os.path.join(repo_root, "datasets", ds_upper, "{}_interpretation_label.txt".format(ds_upper)),
        os.path.join("/root/mtad/datasets/{}".format(dataset), "{}_interpretation_label.txt".format(dataset)),
    ]
    for candidate in candidate_paths:
        if os.path.isfile(candidate):
            return candidate
    return None


def parse_dims(raw):
    tokens = re.findall(r"-?\d+", raw)
    dims = [int(t) for t in tokens]
    return dims


def parse_interpretation_labels(label_path):
    label_ranges = []
    all_dims = []

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        if ":" in line and "-" in line.split(":")[0]:
            left, right = line.split(":", 1)
            start_s, end_s = left.split("-", 1)
            start_i, end_i = int(start_s), int(end_s)
            dims = parse_dims(right)
            label_ranges.append((start_i, end_i, dims))
            all_dims.extend(dims)
        else:
            dims = parse_dims(line)
            if dims:
                label_ranges.append((0, 10 ** 18, dims))
                all_dims.extend(dims)

    return label_ranges, all_dims


def maybe_to_zero_based(dims):
    if not dims:
        return dims
    if min(dims) >= 1:
        return [d - 1 for d in dims]
    return dims


def get_true_dims_for_index(index_i, label_ranges):
    for start_i, end_i, dims in label_ranges:
        if start_i <= index_i <= end_i:
            return dims
    return []


def sensor_scores_from_attention(attn_matrix):
    row_sum = np.sum(attn_matrix, axis=1)
    col_sum = np.sum(attn_matrix, axis=0)
    return row_sum + col_sum


def load_rca_attention(target_dir):
    rca_dir = os.path.join(target_dir, "rca_attention")
    if not os.path.isdir(rca_dir):
        warnings.warn("rca_attention folder not found under {}".format(target_dir))
        return None

    all_attn_path = os.path.join(rca_dir, "all_sparse_attention.npy")
    if not os.path.isfile(all_attn_path):
        warnings.warn("all_sparse_attention.npy not found: {}".format(all_attn_path))
        return None

    all_attn = np.load(all_attn_path)
    if all_attn.ndim != 3 or all_attn.shape[1] != all_attn.shape[2]:
        warnings.warn("Unexpected sparse attention shape: {}".format(all_attn.shape))
        return None
    return all_attn


def normalize_label_ranges(label_ranges, all_dims, n_features):
    normalized = []
    one_based = len(all_dims) > 0 and min(all_dims) >= 1
    for start_i, end_i, dims in label_ranges:
        if one_based:
            dims = maybe_to_zero_based(dims)
        dims = sorted({int(d) for d in dims if 0 <= int(d) < n_features})
        if dims:
            normalized.append((int(start_i), int(end_i), dims))
    return normalized


def build_rca_eval_samples(label_ranges, n_timestamps, eval_unit="timestamp"):
    samples = []
    for start_i, end_i, dims in label_ranges:
        clipped_start = max(0, int(start_i))
        clipped_end = min(int(end_i), n_timestamps - 1)
        if clipped_start > clipped_end or not dims:
            continue

        if eval_unit == "window":
            samples.append(
                {
                    "start": clipped_start,
                    "end": clipped_end,
                    "gt_dims": dims,
                }
            )
        else:
            for idx in range(clipped_start, clipped_end + 1):
                samples.append(
                    {
                        "index": idx,
                        "gt_dims": dims,
                    }
                )
    return samples


def compute_dynamic_k(n_relevant, ratio):
    return max(1, int(np.floor(float(n_relevant) * float(ratio))))


def compute_dcg_at_k(ranked, gt_set, k):
    dcg = 0.0
    for rank_i, dim in enumerate(ranked[:k], start=1):
        if int(dim) in gt_set:
            dcg += 1.0 / np.log2(rank_i + 1.0)
    return dcg


def compute_ndcg_at_k(ranked, gt_set, k):
    ideal_hits = min(len(gt_set), int(k))
    if ideal_hits <= 0:
        return None
    idcg = sum(1.0 / np.log2(rank_i + 1.0) for rank_i in range(1, ideal_hits + 1))
    if idcg <= 0:
        return None
    return compute_dcg_at_k(ranked, gt_set, k) / idcg


def compute_average_precision(ranked, gt_set):
    if not gt_set:
        return None

    hit_count = 0
    precision_sum = 0.0
    for rank_i, dim in enumerate(ranked, start=1):
        if int(dim) in gt_set:
            hit_count += 1
            precision_sum += hit_count / float(rank_i)

    if hit_count == 0:
        return 0.0
    return precision_sum / float(len(gt_set))


def compute_reciprocal_rank(ranked, gt_set):
    for rank_i, dim in enumerate(ranked, start=1):
        if int(dim) in gt_set:
            return 1.0 / float(rank_i)
    return 0.0


def evaluate_ranked_root_causes(ranked, gt_dims):
    gt_set = set(int(d) for d in gt_dims)
    if not gt_set:
        return None

    k100 = compute_dynamic_k(len(gt_set), 1.0)
    k150 = compute_dynamic_k(len(gt_set), 1.5)

    ranked_list = ranked.tolist() if hasattr(ranked, "tolist") else list(ranked)
    metrics = {
        "hit1": 1.0 if len(set(ranked_list[:1]) & gt_set) > 0 else 0.0,
        "hit3": 1.0 if len(set(ranked_list[:3]) & gt_set) > 0 else 0.0,
        "hit5": 1.0 if len(set(ranked_list[:5]) & gt_set) > 0 else 0.0,
        "hitrate_100": len(set(ranked_list[:k100]) & gt_set) / float(len(gt_set)),
        "hitrate_150": len(set(ranked_list[:k150]) & gt_set) / float(len(gt_set)),
        "ndcg_100": compute_ndcg_at_k(ranked_list, gt_set, k100),
        "ndcg_150": compute_ndcg_at_k(ranked_list, gt_set, k150),
        "mrr": compute_reciprocal_rank(ranked_list, gt_set),
        "map": compute_average_precision(ranked_list, gt_set),
    }
    return metrics


def aggregate_metric_lists(metric_lists):
    aggregated = {}
    for metric_name, values in metric_lists.items():
        if len(values) == 0:
            aggregated[metric_name] = None
        else:
            aggregated[metric_name] = float(np.mean(values))
    return aggregated


def compute_rca_ranking_metrics(target_dir, dataset, group, label_path_override=None, eval_unit="timestamp"):
    all_attn = load_rca_attention(target_dir)
    if all_attn is None:
        return None

    group = infer_group_from_target_dir(dataset, group, target_dir)
    label_path = resolve_label_path(dataset, group, label_path_override=label_path_override)
    if label_path is None or not os.path.isfile(label_path):
        warnings.warn("Interpretation label file not found for dataset/group.")
        return None

    label_ranges, all_dims = parse_interpretation_labels(label_path)
    normalized_ranges = normalize_label_ranges(label_ranges, all_dims, n_features=all_attn.shape[1])
    samples = build_rca_eval_samples(normalized_ranges, n_timestamps=all_attn.shape[0], eval_unit=eval_unit)

    if not samples:
        warnings.warn("No aligned interpretation-label samples found for RCA evaluation.")
        return None

    metric_lists = {
        "hit1": [],
        "hit3": [],
        "hit5": [],
        "hitrate_100": [],
        "hitrate_150": [],
        "ndcg_100": [],
        "ndcg_150": [],
        "mrr": [],
        "map": [],
    }

    for sample in samples:
        if eval_unit == "window":
            start_i = sample["start"]
            end_i = sample["end"]
            attn_matrix = np.mean(all_attn[start_i:end_i + 1], axis=0)
        else:
            attn_matrix = all_attn[sample["index"]]

        scores = sensor_scores_from_attention(attn_matrix)
        ranked = np.argsort(scores)[::-1]
        sample_metrics = evaluate_ranked_root_causes(ranked, sample["gt_dims"])
        if sample_metrics is None:
            continue

        for metric_name, metric_value in sample_metrics.items():
            if metric_value is not None:
                metric_lists[metric_name].append(float(metric_value))

    used_samples = len(metric_lists["mrr"])
    if used_samples == 0:
        warnings.warn("No valid interpretation-label samples remained after RCA alignment.")
        return None

    aggregated = aggregate_metric_lists(metric_lists)
    aggregated["sample_count"] = int(used_samples)
    aggregated["feature_count"] = int(all_attn.shape[1])
    aggregated["eval_unit"] = str(eval_unit)
    aggregated["label_path"] = str(label_path)
    aggregated["source"] = "interpretation_label"
    return aggregated


def plot_anomaly_detection(target_dir):
    test_pkl_path = os.path.join(target_dir, "test_output.pkl")
    if not os.path.isfile(test_pkl_path):
        warnings.warn("test_output.pkl not found: {}".format(test_pkl_path))
        return

    df = pd.read_pickle(test_pkl_path)
    if "A_Score_Global" not in df.columns:
        warnings.warn("A_Score_Global missing in test_output.pkl")
        return
    if "A_True_Global" not in df.columns:
        warnings.warn("A_True_Global missing in test_output.pkl")
        return

    scores = df["A_Score_Global"].values
    labels = df["A_True_Global"].values
    x = np.arange(len(scores))

    set_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(x, scores, color="#1f77b4", linewidth=1.0)
    axes[0].set_ylabel("Anomaly Score", fontsize=12)
    axes[0].set_title("MTAD-GAT Anomaly Detection Report", fontsize=14)
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, labels, color="#d62728", linewidth=1.0)
    axes[1].set_ylabel("Ground Truth", fontsize=12)
    axes[1].set_xlabel("Time Index", fontsize=12)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(target_dir, "anomaly_detection_report.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sparse_heatmap(target_dir):
    rca_dir = os.path.join(target_dir, "rca_attention")
    if not os.path.isdir(rca_dir):
        warnings.warn("rca_attention folder not found under {}".format(target_dir))
        return

    all_attn_path = os.path.join(rca_dir, "all_sparse_attention.npy")
    selected_path = os.path.join(rca_dir, "selected_indices.npy")

    if not os.path.isfile(all_attn_path):
        warnings.warn("all_sparse_attention.npy not found: {}".format(all_attn_path))
        return

    all_attn = np.load(all_attn_path)
    if all_attn.ndim != 3 or all_attn.shape[1] != all_attn.shape[2]:
        warnings.warn("Unexpected sparse attention shape: {}".format(all_attn.shape))
        return

    if os.path.isfile(selected_path):
        selected = np.load(selected_path)
        if len(selected) > 0:
            idx = int(selected[0])
        else:
            idx = 0
    else:
        warnings.warn("selected_indices.npy not found, fallback to index 0.")
        idx = 0

    idx = max(0, min(idx, all_attn.shape[0] - 1))
    mat = all_attn[idx]

    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_title("Adaptive Sparse Feature Correlation", fontsize=14)
    ax.set_xlabel("Source Sensor", fontsize=12)
    ax.set_ylabel("Target Sensor", fontsize=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention Weight", rotation=90)
    fig.tight_layout()

    save_path = os.path.join(target_dir, "sparse_attention_heatmap.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_metric(v):
    if v is None:
        return "N/A"
    return "{:.4f}".format(float(v))


def set_publication_style():
    for style_name in ("seaborn-v0_8", "seaborn", "ggplot"):
        try:
            plt.style.use(style_name)
            return
        except Exception:
            continue


def build_markdown_summary(dataset, group, target_dir, f1, precision, recall, rca_metrics=None):
    timestamp = os.path.basename(target_dir.rstrip("/"))
    group_text = group if group is not None else "-"
    lines = []
    lines.append("Detection Summary")
    lines.append("| Dataset | Group | Timestamp | F1-Score | Precision | Recall |")
    lines.append("|---|---|---|---:|---:|---:|")
    lines.append(
        "| {ds} | {gp} | {ts} | {f1} | {pr} | {rc} |".format(
            ds=dataset,
            gp=group_text,
            ts=timestamp,
            f1=format_metric(f1),
            pr=format_metric(precision),
            rc=format_metric(recall),
        )
    )
    if rca_metrics is None:
        lines.append("")
        lines.append("RCA Summary")
        lines.append("Interpretation-label-aligned RCA metrics: N/A")
        return "\n".join(lines)

    lines.append("")
    lines.append("RCA Summary")
    lines.append(
        "| Samples | Eval Unit | Hit@1 | Hit@3 | Hit@5 | HitRate@100% | HitRate@150% | NDCG@100% | NDCG@150% | MRR | MAP |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| {samples} | {unit} | {hit1} | {hit3} | {hit5} | {hr100} | {hr150} | {ndcg100} | {ndcg150} | {mrr} | {mapv} |".format(
            samples=rca_metrics.get("sample_count", "N/A"),
            unit=rca_metrics.get("eval_unit", "N/A"),
            hit1=format_metric(rca_metrics.get("hit1")),
            hit3=format_metric(rca_metrics.get("hit3")),
            hit5=format_metric(rca_metrics.get("hit5")),
            hr100=format_metric(rca_metrics.get("hitrate_100")),
            hr150=format_metric(rca_metrics.get("hitrate_150")),
            ndcg100=format_metric(rca_metrics.get("ndcg_100")),
            ndcg150=format_metric(rca_metrics.get("ndcg_150")),
            mrr=format_metric(rca_metrics.get("mrr")),
            mapv=format_metric(rca_metrics.get("map")),
        )
    )
    lines.append("")
    lines.append("RCA source: interpretation_label")
    lines.append("RCA label file: {}".format(rca_metrics.get("label_path", "N/A")))
    return "\n".join(lines)


def main():
    args = parse_args()
    dataset = args.dataset
    group = args.group

    try:
        if args.target_dir is not None:
            target_dir = args.target_dir
            if not os.path.isdir(target_dir):
                raise FileNotFoundError("Provided target_dir does not exist: {}".format(target_dir))
        else:
            base_dir = get_base_dir(args.output_root, dataset, group)
            target_dir = get_latest_target_dir(base_dir)
    except Exception as e:
        print("Error: {}".format(e))
        return

    group = infer_group_from_target_dir(dataset, group, target_dir)
    f1, precision, recall = load_summary_metrics(target_dir, threshold_method=args.threshold_method)

    try:
        rca_metrics = compute_rca_ranking_metrics(
            target_dir,
            dataset,
            group,
            label_path_override=args.label_path,
            eval_unit=args.rca_eval_unit,
        )
    except Exception as e:
        warnings.warn("RCA metric computation failed: {}".format(e))
        rca_metrics = None

    try:
        plot_anomaly_detection(target_dir)
    except Exception as e:
        warnings.warn("plot_anomaly_detection failed: {}".format(e))

    try:
        plot_sparse_heatmap(target_dir)
    except Exception as e:
        warnings.warn("plot_sparse_heatmap failed: {}".format(e))

    md_table = build_markdown_summary(
        dataset=dataset,
        group=group,
        target_dir=target_dir,
        f1=f1,
        precision=precision,
        recall=recall,
        rca_metrics=rca_metrics,
    )

    print(md_table)
    report_path = os.path.join(target_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(md_table + "\n")
    unit_report_path = os.path.join(target_dir, "summary_report_{}.txt".format(args.rca_eval_unit))
    with open(unit_report_path, "w") as f:
        f.write(md_table + "\n")

    if rca_metrics is not None:
        print("HitRate@100%: {}".format(format_metric(rca_metrics.get("hitrate_100"))))
        print("HitRate@150%: {}".format(format_metric(rca_metrics.get("hitrate_150"))))
        print("NDCG@100%: {}".format(format_metric(rca_metrics.get("ndcg_100"))))
        print("NDCG@150%: {}".format(format_metric(rca_metrics.get("ndcg_150"))))
        print("MRR: {}".format(format_metric(rca_metrics.get("mrr"))))
        print("MAP: {}".format(format_metric(rca_metrics.get("map"))))
    else:
        print("RCA metrics: N/A")


if __name__ == "__main__":
    main()
