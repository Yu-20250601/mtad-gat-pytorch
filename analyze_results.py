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
    return parser.parse_args()


def get_base_dir(dataset, group):
    if group:
        return os.path.join("output", dataset, group)
    return os.path.join("output", dataset)


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


def load_summary_metrics(target_dir):
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
            epsilon = summary.get("epsilon_result", {})
            return epsilon.get("f1"), epsilon.get("precision"), epsilon.get("recall")
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
    # Supports comma/space/newline-separated integers
    tokens = re.findall(r"-?\d+", raw)
    dims = [int(t) for t in tokens]
    return dims


def parse_interpretation_labels(label_path):
    """
    Returns:
        label_ranges: list of (start, end, [dims...]) with inclusive range.
        all_dims: flattened list of dims, for 1-based check.
    """
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
            # Fallback: if no range format, keep as global dims
            dims = parse_dims(line)
            if dims:
                label_ranges.append((0, 10 ** 18, dims))
                all_dims.extend(dims)

    return label_ranges, all_dims


def maybe_to_zero_based(dims):
    if not dims:
        return dims
    # If all dims >= 1, likely 1-based indexing.
    if min(dims) >= 1:
        return [d - 1 for d in dims]
    return dims


def get_true_dims_for_index(index_i, label_ranges):
    for start_i, end_i, dims in label_ranges:
        if start_i <= index_i <= end_i:
            return dims
    return []


def sensor_scores_from_attention(attn_matrix):
    # Aggregate inbound + outbound to reflect sensor centrality in sparse graph
    row_sum = np.sum(attn_matrix, axis=1)
    col_sum = np.sum(attn_matrix, axis=0)
    return row_sum + col_sum


def compute_rca_hitk(target_dir, dataset, group, label_path_override=None):
    rca_dir = os.path.join(target_dir, "rca_attention")
    if not os.path.isdir(rca_dir):
        warnings.warn("rca_attention folder not found under {}".format(target_dir))
        return None, None, None

    selected_path = os.path.join(rca_dir, "selected_indices.npy")
    all_attn_path = os.path.join(rca_dir, "all_sparse_attention.npy")

    if not os.path.isfile(selected_path):
        warnings.warn("selected_indices.npy not found: {}".format(selected_path))
        return None, None, None
    if not os.path.isfile(all_attn_path):
        warnings.warn("all_sparse_attention.npy not found: {}".format(all_attn_path))
        return None, None, None

    group = infer_group_from_target_dir(dataset, group, target_dir)
    label_path = resolve_label_path(dataset, group, label_path_override=label_path_override)
    if label_path is None or not os.path.isfile(label_path):
        warnings.warn("Interpretation label file not found for dataset/group.")
        return None, None, None

    selected = np.load(selected_path)
    all_attn = np.load(all_attn_path)
    label_ranges, all_dims = parse_interpretation_labels(label_path)

    # Auto align dimensional indexing
    one_based = len(all_dims) > 0 and min(all_dims) >= 1
    if one_based:
        label_ranges = [(s, e, maybe_to_zero_based(d)) for s, e, d in label_ranges]

    hit1_list = []
    hit3_list = []
    hit5_list = []

    for idx in selected:
        idx = int(idx)
        if idx < 0 or idx >= all_attn.shape[0]:
            continue

        gt_dims = get_true_dims_for_index(idx, label_ranges)
        if not gt_dims:
            continue

        scores = sensor_scores_from_attention(all_attn[idx])
        ranked = np.argsort(scores)[::-1]

        top1 = set(ranked[:1].tolist())
        top3 = set(ranked[:3].tolist())
        top5 = set(ranked[:5].tolist())
        gt_set = set(gt_dims)

        hit1_list.append(1.0 if len(top1 & gt_set) > 0 else 0.0)
        hit3_list.append(1.0 if len(top3 & gt_set) > 0 else 0.0)
        hit5_list.append(1.0 if len(top5 & gt_set) > 0 else 0.0)

    if not hit1_list:
        warnings.warn("No aligned samples between selected_indices and interpretation labels.")
        return None, None, None

    return float(np.mean(hit1_list)), float(np.mean(hit3_list)), float(np.mean(hit5_list))


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


def build_markdown_summary(dataset, group, target_dir, f1, precision, recall, hit1, hit3):
    timestamp = os.path.basename(target_dir.rstrip("/"))
    group_text = group if group is not None else "-"
    table = []
    table.append("| Dataset | Group | Timestamp | F1-Score | Precision | Recall | Hit@1 | Hit@3 |")
    table.append("|---|---|---|---:|---:|---:|---:|---:|")
    table.append(
        "| {ds} | {gp} | {ts} | {f1} | {pr} | {rc} | {h1} | {h3} |".format(
            ds=dataset,
            gp=group_text,
            ts=timestamp,
            f1=format_metric(f1),
            pr=format_metric(precision),
            rc=format_metric(recall),
            h1=format_metric(hit1),
            h3=format_metric(hit3),
        )
    )
    return "\n".join(table)


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
            base_dir = get_base_dir(dataset, group)
            target_dir = get_latest_target_dir(base_dir)
    except Exception as e:
        print("Error: {}".format(e))
        return

    group = infer_group_from_target_dir(dataset, group, target_dir)
    f1, precision, recall = load_summary_metrics(target_dir)

    try:
        hit1, hit3, hit5 = compute_rca_hitk(
            target_dir,
            dataset,
            group,
            label_path_override=args.label_path,
        )
    except Exception as e:
        warnings.warn("RCA metric computation failed: {}".format(e))
        hit1, hit3, hit5 = None, None, None

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
        hit1=hit1,
        hit3=hit3,
    )

    print(md_table)
    report_path = os.path.join(target_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(md_table + "\n")

    # Additional metric is computed by requirement but not required in output table.
    if hit5 is not None:
        print("Hit@5: {:.4f}".format(hit5))
    else:
        print("Hit@5: N/A")


if __name__ == "__main__":
    main()

