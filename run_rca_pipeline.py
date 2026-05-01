import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_results import compute_rca_ranking_metrics, load_summary_metrics


def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def bool_to_cli(value: bool) -> str:
    return "true" if value else "false"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(command: List[str], cwd: Path, log_path: Path) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND:\n")
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(command)}")


def model_to_train_args(model_name: str) -> List[str]:
    mapping = {
        "mtad_gat_sparsemax_vae": [
            "--use_adaptive_sparse_feat_gat",
            "false",
            "--use_node_embedding",
            "false",
            "--use_sparsemax",
            "true",
            "--recon_model",
            "vae",
            "--vae_latent_dim",
            "64",
            "--kl_beta",
            "0.001",
            "--export_sparse_attention",
            "true",
            "--sparse_attention_topk",
            "50",
        ],
        "mtad_gat_sparsemax_vae_node": [
            "--use_adaptive_sparse_feat_gat",
            "true",
            "--use_node_embedding",
            "true",
            "--node_embed_dim",
            "16",
            "--use_sparsemax",
            "true",
            "--recon_model",
            "vae",
            "--vae_latent_dim",
            "64",
            "--kl_beta",
            "0.001",
            "--export_sparse_attention",
            "true",
            "--sparse_attention_topk",
            "50",
        ],
    }
    if model_name not in mapping:
        raise KeyError(f"Unsupported RCA model: {model_name}")
    return mapping[model_name]


def run_detailed_train(
    repo_root: Path,
    artifact_root: Path,
    group: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    use_gatv2: bool,
) -> Path:
    group_root = repo_root / "output" / "SMD" / group
    ensure_dir(group_root)
    before = {path.resolve() for path in group_root.iterdir() if path.is_dir()}
    command = [
        sys.executable,
        "train.py",
        "--dataset",
        "SMD",
        "--group",
        group,
        "--lookback",
        "100",
        "--epochs",
        str(epochs),
        "--bs",
        str(batch_size),
        "--use_gatv2",
        bool_to_cli(use_gatv2),
        "--rca_threshold_method",
        "pot",
    ] + model_to_train_args(model_name)
    safe_name = f"train_{group}_{model_name}.log"
    run_command(command, repo_root, artifact_root / "logs" / safe_name)

    after = [path.resolve() for path in group_root.iterdir() if path.is_dir()]
    new_dirs = [path for path in after if path not in before and path.name != "logs"]
    if new_dirs:
        return max(new_dirs, key=lambda item: item.stat().st_mtime)

    candidates = [path for path in after if path.name != "logs"]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {group_root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def analyze_run(repo_root: Path, artifact_root: Path, group: str, run_dir: Path, eval_unit: str) -> None:
    command = [
        sys.executable,
        "analyze_results.py",
        "--dataset",
        "SMD",
        "--group",
        group,
        "--target_dir",
        str(run_dir),
        "--threshold_method",
        "pot_result",
        "--rca_eval_unit",
        eval_unit,
    ]
    safe_name = f"analyze_{group}_{run_dir.name}_{eval_unit}.log"
    run_command(command, repo_root, artifact_root / "logs" / safe_name)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [headers, ["---"] * len(headers)]
    for _, row in df.iterrows():
        rows.append([str(row[col]) for col in headers])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def save_table(df: pd.DataFrame, output_csv: Path, output_md: Path) -> None:
    df.to_csv(output_csv, index=False)
    output_md.write_text(dataframe_to_markdown(df), encoding="utf-8")


def build_metric_table(df: pd.DataFrame, metric: str, eval_unit: str) -> pd.DataFrame:
    filtered = df[df["eval_unit"] == eval_unit].copy()
    pivot = filtered.pivot_table(index="model", columns="group", values=metric, aggfunc="first").reset_index()
    group_cols = [col for col in pivot.columns if col != "model"]
    pivot["Average"] = pivot[group_cols].mean(axis=1)
    return pivot[["model"] + group_cols + ["Average"]].sort_values("Average", ascending=False).reset_index(drop=True)


def plot_grouped_bar(df: pd.DataFrame, metric: str, eval_unit: str, output_path: Path) -> None:
    filtered = df[df["eval_unit"] == eval_unit].copy()
    pivot = filtered.pivot_table(index="group", columns="model", values=metric, aggfunc="first")
    plt.style.use("ggplot")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{eval_unit} {metric.upper()} by machine")
    ax.set_ylim(0, 1.05)
    ax.legend(title="model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_average_bar(df: pd.DataFrame, metric: str, eval_unit: str, output_path: Path) -> None:
    table = build_metric_table(df, metric, eval_unit)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    ax.bar(table["model"], table["Average"], color=colors[: len(table)])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Average {eval_unit} {metric.upper()} by model")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def collect_result_rows(detailed_runs: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group, model_runs in detailed_runs.items():
        for model_name, run_dir_text in model_runs.items():
            run_dir = Path(run_dir_text)
            f1, precision, recall = load_summary_metrics(str(run_dir), threshold_method="pot_result")
            for eval_unit in ("timestamp", "window"):
                metrics = compute_rca_ranking_metrics(str(run_dir), "SMD", group, eval_unit=eval_unit)
                if metrics is None:
                    metrics = {}
                rows.append(
                    {
                        "group": group,
                        "model": model_name,
                        "run_dir": str(run_dir),
                        "eval_unit": eval_unit,
                        "detection_f1": f1,
                        "detection_precision": precision,
                        "detection_recall": recall,
                        "sample_count": metrics.get("sample_count"),
                        "hit1": metrics.get("hit1"),
                        "hit3": metrics.get("hit3"),
                        "hit5": metrics.get("hit5"),
                        "hitrate_100": metrics.get("hitrate_100"),
                        "hitrate_150": metrics.get("hitrate_150"),
                        "ndcg_100": metrics.get("ndcg_100"),
                        "ndcg_150": metrics.get("ndcg_150"),
                        "mrr": metrics.get("mrr"),
                        "map": metrics.get("map"),
                    }
                )
    return pd.DataFrame(rows)


def load_selected_attention(run_dir: Path) -> np.ndarray:
    rca_dir = run_dir / "rca_attention"
    all_attn = np.load(rca_dir / "all_sparse_attention.npy")
    selected_path = rca_dir / "selected_indices.npy"
    if selected_path.exists():
        selected = np.load(selected_path)
        idx = int(selected[0]) if len(selected) > 0 else 0
    else:
        idx = 0
    idx = max(0, min(idx, all_attn.shape[0] - 1))
    return all_attn[idx]


def render_heatmap_comparison(run_dirs: Dict[str, str], output_path: Path, title: str) -> None:
    model_items = list(run_dirs.items())
    mats = [(model_name, load_selected_attention(Path(run_dir))) for model_name, run_dir in model_items]
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, len(mats), figsize=(6.5 * len(mats), 5), constrained_layout=True)
    if len(mats) == 1:
        axes = [axes]

    for ax, (model_name, mat) in zip(axes, mats):
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_title(model_name)
        ax.set_xlabel("Source sensor")
        ax.set_ylabel("Target sensor")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_case_figure(repo_root: Path, artifact_root: Path, run_dirs: Dict[str, str], group: str) -> None:
    labels = list(run_dirs.keys())
    dirs = list(run_dirs.values())
    command = [
        sys.executable,
        "render_case_figures.py",
        "--run_dirs",
    ] + dirs + [
        "--labels",
        ",".join(labels),
        "--threshold_method",
        "pot_result",
        "--output",
        str(artifact_root / "paper_assets" / "figures" / f"rca_case_compare_{group}.png"),
        "--title",
        f"SMD {group} anomaly/RCA comparison (POT)",
        "--context",
        "250",
    ]
    run_command(command, repo_root, artifact_root / "logs" / f"case_compare_{group}.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MTAD-GAT RCA experiments and build paper-ready tables/figures.")
    parser.add_argument("--groups", type=parse_list_arg, default=["1-2", "2-1", "3-2"])
    parser.add_argument(
        "--models",
        type=parse_list_arg,
        default=["mtad_gat_sparsemax_vae", "mtad_gat_sparsemax_vae_node"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_gatv2", type=str, default="true")
    parser.add_argument("--output_root", type=str, default="rca_artifacts")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    artifact_root = (repo_root / args.output_root).resolve()
    tables_dir = artifact_root / "paper_assets" / "tables"
    figures_dir = artifact_root / "paper_assets" / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(artifact_root / "logs")

    use_gatv2 = args.use_gatv2.lower() in {"1", "true", "yes", "y", "t"}
    detailed_runs: Dict[str, Dict[str, str]] = {}

    for group in args.groups:
        detailed_runs[group] = {}
        for model_name in args.models:
            run_dir = run_detailed_train(
                repo_root=repo_root,
                artifact_root=artifact_root,
                group=group,
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                use_gatv2=use_gatv2,
            )
            analyze_run(repo_root, artifact_root, group, run_dir, "timestamp")
            analyze_run(repo_root, artifact_root, group, run_dir, "window")
            detailed_runs[group][model_name] = str(run_dir)

        render_case_figure(repo_root, artifact_root, detailed_runs[group], group)
        render_heatmap_comparison(
            detailed_runs[group],
            figures_dir / f"rca_heatmap_compare_{group}.png",
            title=f"SMD {group} sparse attention heatmap comparison",
        )

    results_df = collect_result_rows(detailed_runs)
    save_table(results_df, tables_dir / "rca_results_all.csv", tables_dir / "rca_results_all.md")

    for eval_unit in ("timestamp", "window"):
        for metric in ("hit1", "hit3", "hit5", "mrr", "map"):
            table = build_metric_table(results_df, metric, eval_unit)
            save_table(
                table,
                tables_dir / f"rca_{eval_unit}_{metric}.csv",
                tables_dir / f"rca_{eval_unit}_{metric}.md",
            )
            plot_grouped_bar(results_df, metric, eval_unit, figures_dir / f"rca_{eval_unit}_{metric}_by_machine.png")
            plot_average_bar(results_df, metric, eval_unit, figures_dir / f"rca_{eval_unit}_{metric}_average.png")

    manifest = {
        "groups": args.groups,
        "models": args.models,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "use_gatv2": use_gatv2,
        "detailed_runs": detailed_runs,
        "tables": sorted(str(path) for path in tables_dir.glob("*")),
        "figures": sorted(str(path) for path in figures_dir.glob("*")),
    }
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RCA pipeline completed. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
