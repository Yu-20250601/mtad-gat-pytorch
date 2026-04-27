import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from eval_methods import adjust_predicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render publication-style anomaly case figures from saved MTAD-GAT outputs."
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="One or more run directories containing summary*.txt and test_output.pkl.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional comma-separated labels for the run_dirs.",
    )
    parser.add_argument(
        "--threshold_method",
        type=str,
        default="bf_result",
        choices=["bf_result", "epsilon_result", "pot_result"],
        help="Thresholding result used for metrics, threshold line, and predicted anomaly intervals.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the saved PNG figure.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--zoom_start",
        type=int,
        default=None,
        help="Optional manual zoom window start index.",
    )
    parser.add_argument(
        "--zoom_end",
        type=int,
        default=None,
        help="Optional manual zoom window end index.",
    )
    parser.add_argument(
        "--zoom_mode",
        type=str,
        default="longest_true",
        choices=["longest_true", "highest_score"],
        help="Automatic zoom selection strategy when zoom_start/zoom_end are not provided.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=250,
        help="Context length added before and after the selected zoom event.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved figure DPI.",
    )
    return parser.parse_args()


def set_publication_style() -> None:
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-v0_8", "seaborn", "ggplot"):
        try:
            plt.style.use(style_name)
            break
        except Exception:
            continue
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def parse_labels(raw: str, count: int) -> List[str]:
    if raw is None:
        return []
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    if len(labels) != count:
        raise ValueError(
            f"--labels provided {len(labels)} entries, but {count} run directories were given."
        )
    return labels


def load_latest_summary(run_dir: Path) -> Dict[str, Dict[str, float]]:
    summary_files = sorted(run_dir.glob("summary*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not summary_files:
        raise FileNotFoundError(f"No summary*.txt found under {run_dir}")

    for summary_path in summary_files:
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue

    raise ValueError(f"No valid summary JSON found under {run_dir}")


def load_run_label(run_dir: Path) -> str:
    config_path = run_dir / "config.txt"
    if not config_path.exists():
        return run_dir.name

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        group = config.get("group", "?")
        recon_model = str(config.get("recon_model", "gru")).lower()
        use_gatv2 = bool(config.get("use_gatv2", True))
        model_name = f"MTAD-GAT-{recon_model.upper()}"
        if use_gatv2:
            model_name += " (GATv2)"
        return f"{group} | {model_name}"
    except Exception:
        return run_dir.name


def find_intervals(values: Sequence[int]) -> List[Tuple[int, int]]:
    arr = np.asarray(values).astype(int)
    if arr.size == 0:
        return []

    changes = np.where(arr[1:] != arr[:-1])[0] + 1
    if arr[0] == 1:
        changes = np.insert(changes, 0, 0)

    intervals: List[Tuple[int, int]] = []
    for idx in range(0, len(changes) - 1, 2):
        intervals.append((int(changes[idx]), int(changes[idx + 1] - 1)))
    if len(changes) % 2 == 1:
        intervals.append((int(changes[-1]), int(len(arr) - 1)))
    return intervals


def choose_zoom_window(
    labels: np.ndarray,
    scores: np.ndarray,
    zoom_start: int,
    zoom_end: int,
    zoom_mode: str,
    context: int,
) -> Tuple[int, int]:
    n = len(scores)
    if zoom_start is not None and zoom_end is not None:
        start = max(0, min(int(zoom_start), n - 1))
        end = max(start + 1, min(int(zoom_end), n))
        return start, end

    if labels is not None and np.sum(labels) > 0 and zoom_mode == "longest_true":
        true_intervals = find_intervals(labels)
        target_start, target_end = max(true_intervals, key=lambda item: item[1] - item[0])
        start = max(0, target_start - context)
        end = min(n, target_end + context + 1)
        return start, end

    peak_idx = int(np.argmax(scores))
    start = max(0, peak_idx - context)
    end = min(n, peak_idx + context + 1)
    return start, end


def make_predictions(scores: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    raw_pred = (scores >= threshold).astype(int)
    if labels is None:
        return raw_pred
    adjusted = adjust_predicts(None, labels.astype(int), threshold, pred=raw_pred.copy(), calc_latency=False)
    return np.asarray(adjusted).astype(int)


def format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def add_interval_shading(ax, intervals: Sequence[Tuple[int, int]], color: str, alpha: float, label: str) -> None:
    first = True
    for start, end in intervals:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0, label=label if first else None)
        first = False


def plot_score_panel(
    ax,
    scores: np.ndarray,
    threshold: float,
    true_intervals: Sequence[Tuple[int, int]],
    pred_intervals: Sequence[Tuple[int, int]],
    xlim: Tuple[int, int],
    panel_title: str,
    metric_text: str,
) -> None:
    start, end = xlim
    x = np.arange(len(scores))
    y_slice = scores[start:end]
    y_min = float(np.min(y_slice))
    y_max = float(np.max(y_slice))
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_pad = 0.08 * (y_max - y_min)
    ax.set_xlim(start, end - 1)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    add_interval_shading(ax, pred_intervals, color="#4c78a8", alpha=0.18, label="Predicted anomaly")
    add_interval_shading(ax, true_intervals, color="#e45756", alpha=0.18, label="Ground-truth anomaly")

    ax.plot(x, scores, color="#1f77b4", linewidth=1.1, label="Anomaly score")
    ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1.0, label="Threshold")
    ax.set_title(panel_title)
    ax.set_ylabel("Score")
    ax.grid(alpha=0.25)
    ax.text(
        0.01,
        0.98,
        metric_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
    )


def build_metric_text(method_name: str, metrics: Dict[str, float]) -> str:
    title_map = {
        "bf_result": "BF",
        "epsilon_result": "Epsilon",
        "pot_result": "POT",
    }
    return (
        f"{title_map[method_name]} | "
        f"F1={format_metric(metrics['f1'])}  "
        f"P={format_metric(metrics['precision'])}  "
        f"R={format_metric(metrics['recall'])}"
    )


def build_legend_handles() -> List[object]:
    return [
        Line2D([0], [0], color="#1f77b4", lw=1.2, label="Anomaly score"),
        Line2D([0], [0], color="#222222", lw=1.0, linestyle="--", label="Threshold"),
        Patch(facecolor="#4c78a8", edgecolor="none", alpha=0.18, label="Predicted anomaly"),
        Patch(facecolor="#e45756", edgecolor="none", alpha=0.18, label="Ground-truth anomaly"),
    ]


def build_run_payload(run_dir: Path, threshold_method: str, custom_label: str = None) -> Dict[str, object]:
    summary = load_latest_summary(run_dir)
    if threshold_method not in summary:
        raise KeyError(f"{threshold_method} not found in {run_dir}")

    output_path = run_dir / "test_output.pkl"
    if not output_path.exists():
        raise FileNotFoundError(f"test_output.pkl not found under {run_dir}")

    df = pd.read_pickle(output_path)
    if "A_Score_Global" not in df.columns:
        raise KeyError(f"A_Score_Global missing in {output_path}")

    scores = df["A_Score_Global"].to_numpy(dtype=float)
    labels = df["A_True_Global"].to_numpy(dtype=int) if "A_True_Global" in df.columns else None
    metrics = summary[threshold_method]
    threshold = float(metrics["threshold"])
    preds = make_predictions(scores, labels, threshold)

    return {
        "run_dir": run_dir,
        "label": custom_label or load_run_label(run_dir),
        "scores": scores,
        "labels": labels,
        "preds": preds,
        "metrics": metrics,
        "threshold": threshold,
    }


def render_single_figure(
    payload: Dict[str, object],
    threshold_method: str,
    zoom_range: Tuple[int, int],
    output_path: Path,
    title: str = None,
    dpi: int = 300,
) -> None:
    scores = payload["scores"]
    labels = payload["labels"]
    preds = payload["preds"]
    threshold = payload["threshold"]
    metrics = payload["metrics"]
    label = payload["label"]

    true_intervals = find_intervals(labels) if labels is not None else []
    pred_intervals = find_intervals(preds)
    zoom_start, zoom_end = zoom_range
    zoom_true = [(max(s, zoom_start), min(e, zoom_end - 1)) for s, e in true_intervals if e >= zoom_start and s < zoom_end]
    zoom_pred = [(max(s, zoom_start), min(e, zoom_end - 1)) for s, e in pred_intervals if e >= zoom_start and s < zoom_end]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(title or label, fontsize=15)

    plot_score_panel(
        axes[0],
        scores,
        threshold,
        true_intervals,
        pred_intervals,
        xlim=(0, len(scores)),
        panel_title="Global anomaly score",
        metric_text=build_metric_text(threshold_method, metrics),
    )
    plot_score_panel(
        axes[1],
        scores,
        threshold,
        zoom_true,
        zoom_pred,
        xlim=(zoom_start, zoom_end),
        panel_title=f"Zoomed view ({zoom_start}:{zoom_end})",
        metric_text=f"Threshold={format_metric(threshold)}",
    )
    axes[1].set_xlabel("Time index")

    fig.legend(handles=build_legend_handles(), loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.02))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_comparison_figure(
    payloads: Sequence[Dict[str, object]],
    threshold_method: str,
    zoom_range: Tuple[int, int],
    output_path: Path,
    title: str = None,
    dpi: int = 300,
) -> None:
    zoom_start, zoom_end = zoom_range
    n_rows = len(payloads)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4.6 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray([axes])
    fig.suptitle(title or "Case comparison", fontsize=15)

    for row_idx, payload in enumerate(payloads):
        scores = payload["scores"]
        labels = payload["labels"]
        preds = payload["preds"]
        threshold = payload["threshold"]
        metrics = payload["metrics"]
        label = payload["label"]

        true_intervals = find_intervals(labels) if labels is not None else []
        pred_intervals = find_intervals(preds)
        zoom_true = [(max(s, zoom_start), min(e, zoom_end - 1)) for s, e in true_intervals if e >= zoom_start and s < zoom_end]
        zoom_pred = [(max(s, zoom_start), min(e, zoom_end - 1)) for s, e in pred_intervals if e >= zoom_start and s < zoom_end]

        plot_score_panel(
            axes[row_idx, 0],
            scores,
            threshold,
            true_intervals,
            pred_intervals,
            xlim=(0, len(scores)),
            panel_title=f"{label} | Global",
            metric_text=build_metric_text(threshold_method, metrics),
        )
        plot_score_panel(
            axes[row_idx, 1],
            scores,
            threshold,
            zoom_true,
            zoom_pred,
            xlim=(zoom_start, zoom_end),
            panel_title=f"{label} | Zoom ({zoom_start}:{zoom_end})",
            metric_text=f"Threshold={format_metric(threshold)}",
        )

    axes[-1, 0].set_xlabel("Time index")
    axes[-1, 1].set_xlabel("Time index")
    fig.legend(handles=build_legend_handles(), loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.02))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_publication_style()

    run_dirs = [Path(item).expanduser().resolve() for item in args.run_dirs]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    custom_labels = parse_labels(args.labels, len(run_dirs))
    payloads = []
    for idx, run_dir in enumerate(run_dirs):
        custom_label = custom_labels[idx] if custom_labels else None
        payloads.append(build_run_payload(run_dir, args.threshold_method, custom_label=custom_label))

    reference = payloads[0]
    zoom_range = choose_zoom_window(
        labels=reference["labels"],
        scores=reference["scores"],
        zoom_start=args.zoom_start,
        zoom_end=args.zoom_end,
        zoom_mode=args.zoom_mode,
        context=args.context,
    )

    output_path = Path(args.output).expanduser().resolve()
    if len(payloads) == 1:
        render_single_figure(
            payload=payloads[0],
            threshold_method=args.threshold_method,
            zoom_range=zoom_range,
            output_path=output_path,
            title=args.title,
            dpi=args.dpi,
        )
    else:
        render_comparison_figure(
            payloads=payloads,
            threshold_method=args.threshold_method,
            zoom_range=zoom_range,
            output_path=output_path,
            title=args.title,
            dpi=args.dpi,
        )

    print(f"Saved figure to {output_path}")
    print(f"Zoom window: {zoom_range[0]}:{zoom_range[1]}")


if __name__ == "__main__":
    main()
