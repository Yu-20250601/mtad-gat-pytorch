import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_results(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "benchmark_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing benchmark_results.csv: {csv_path}")
    return pd.read_csv(csv_path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [headers, ["---"] * len(headers)]
    for _, row in df.iterrows():
        rows.append([str(row[col]) for col in headers])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def build_metric_table(df: pd.DataFrame, threshold_method: str, metric: str) -> pd.DataFrame:
    filtered = df[df["threshold_method"] == threshold_method].copy()
    pivot = (
        filtered.pivot_table(index="model", columns="entity", values=metric, aggfunc="first")
        .reset_index()
    )
    entity_cols = [col for col in pivot.columns if col != "model"]
    pivot["Average"] = pivot[entity_cols].mean(axis=1)
    ordered_cols = ["model"] + entity_cols + ["Average"]
    return pivot[ordered_cols].sort_values("Average", ascending=False).reset_index(drop=True)


def save_table(df: pd.DataFrame, output_csv: Path, output_md: Path) -> None:
    df.to_csv(output_csv, index=False)
    output_md.write_text(dataframe_to_markdown(df), encoding="utf-8")


def plot_average_bar(df: pd.DataFrame, value_col: str, title: str, output_path: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["model"], df[value_col], color=["#4c78a8", "#f58518", "#54a24b", "#e45756"][: len(df)])
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, max(0.1, df[value_col].max() * 1.15)))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bar(df: pd.DataFrame, threshold_method: str, metric: str, output_path: Path) -> None:
    filtered = df[df["threshold_method"] == threshold_method].copy()
    pivot = filtered.pivot_table(index="entity", columns="model", values=metric, aggfunc="first")
    plt.style.use("ggplot")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel(metric)
    ax.set_title(f"{threshold_method} {metric} by machine")
    ax.set_ylim(0, 1.05)
    ax.legend(title="model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_aux_average_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["model", "threshold_method"], as_index=False)[["f1", "precision", "recall"]]
        .mean()
        .sort_values(["threshold_method", "f1"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def build_seed_stability_table(seed_dirs: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run_dir in seed_dirs:
        df = load_results(run_dir)
        filtered = df[df["threshold_method"] == "pot_result"].copy()
        grouped = filtered.groupby("model", as_index=False)[["f1", "precision", "recall"]].mean()
        grouped["run_dir"] = str(run_dir)
        rows.extend(grouped.to_dict("records"))

    merged = pd.DataFrame(rows)
    summary = (
        merged.groupby("model", as_index=False)[["f1", "precision", "recall"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "model",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
    ]
    return summary.sort_values("f1_mean", ascending=False).reset_index(drop=True)


def build_gatv2_table(true_run: Path, false_run: Path) -> pd.DataFrame:
    rows = []
    for label, run_dir in [("use_gatv2=true", true_run), ("use_gatv2=false", false_run)]:
        df = load_results(run_dir)
        filtered = df[df["threshold_method"] == "pot_result"].copy()
        grouped = filtered.groupby("model", as_index=False)[["f1", "precision", "recall"]].mean()
        grouped.insert(0, "setting", label)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate thesis benchmark outputs into tables and figures.")
    parser.add_argument("--main_run", required=True, type=str, help="Main benchmark run directory.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory for aggregated tables and figures.")
    parser.add_argument("--seed_runs", type=parse_list_arg, default=[], help="Optional comma-separated seed run directories.")
    parser.add_argument("--gatv2_true_run", type=str, default=None)
    parser.add_argument("--gatv2_false_run", type=str, default=None)
    args = parser.parse_args()

    main_run = Path(args.main_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    main_df = load_results(main_run)

    for metric in ["f1", "precision", "recall"]:
        table = build_metric_table(main_df, "pot_result", metric)
        save_table(
            table,
            tables_dir / f"main_pot_{metric}.csv",
            tables_dir / f"main_pot_{metric}.md",
        )

    aux_table = build_aux_average_table(main_df)
    save_table(
        aux_table,
        tables_dir / "aux_threshold_averages.csv",
        tables_dir / "aux_threshold_averages.md",
    )

    main_pot_f1 = build_metric_table(main_df, "pot_result", "f1")
    plot_average_bar(main_pot_f1, "Average", "Average POT F1 by model", figures_dir / "pot_average_f1.png")
    plot_grouped_bar(main_df, "pot_result", "f1", figures_dir / "pot_f1_by_machine.png")

    if args.seed_runs:
        seed_dirs = [Path(item).resolve() for item in args.seed_runs]
        seed_table = build_seed_stability_table(seed_dirs)
        save_table(
            seed_table,
            tables_dir / "seed_stability_pot.csv",
            tables_dir / "seed_stability_pot.md",
        )

    if args.gatv2_true_run and args.gatv2_false_run:
        gatv2_table = build_gatv2_table(Path(args.gatv2_true_run).resolve(), Path(args.gatv2_false_run).resolve())
        save_table(
            gatv2_table,
            tables_dir / "gatv2_ablation_pot.csv",
            tables_dir / "gatv2_ablation_pot.md",
        )

    manifest = {
        "main_run": str(main_run),
        "output_dir": str(output_dir),
        "generated_tables": sorted(str(path) for path in tables_dir.glob("*")),
        "generated_figures": sorted(str(path) for path in figures_dir.glob("*")),
    }
    (output_dir / "manifest.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in manifest.items()),
        encoding="utf-8",
    )
    print(f"Saved thesis tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
