import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("布尔参数只接受 true/false、yes/no、1/0 等形式。")


def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="筛选论文主模型的辅助脚本。")
    parser.add_argument("--run_dir", type=str, default=None, help="已有 benchmark 结果目录；提供后将跳过重新运行。")
    parser.add_argument("--datasets", type=parse_list_arg, default=["SMD"])
    parser.add_argument("--smd_groups", type=parse_list_arg, default=["1-1"])
    parser.add_argument(
        "--models",
        type=parse_list_arg,
        default=[
            "mtad_gat_gru",
            "mtad_gat_vae",
            "mtad_gat_sparsemax_vae",
            "mtad_gat_sparsemax_vae_node",
        ],
    )
    parser.add_argument("--selection_root", type=str, default="model_selection_output")
    parser.add_argument("--benchmark_output_root", type=str, default="benchmark_output")
    parser.add_argument(
        "--primary_selector",
        type=str,
        default="mean_rank",
        choices=["mean_rank", "mean_f1", "bf_f1", "pot_f1", "epsilon_f1"],
        help="主模型排序规则。",
    )

    # 下面这些参数会透传给 benchmark_models.py
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--max_train_size", type=int, default=4000)
    parser.add_argument("--max_test_size", type=int, default=20000)
    parser.add_argument("--mtad_epochs", type=int, default=5)
    parser.add_argument("--mtad_lr", type=float, default=1e-3)
    parser.add_argument("--mtad_gru_hid_dim", type=int, default=150)
    parser.add_argument("--mtad_fc_hid_dim", type=int, default=150)
    parser.add_argument("--mtad_recon_hid_dim", type=int, default=150)
    parser.add_argument("--mtad_node_embed_dim", type=int, default=16)
    parser.add_argument("--mtad_vae_latent_dim", type=int, default=64)
    parser.add_argument("--mtad_kl_beta", type=float, default=0.001)
    return parser


def bool_to_cli(value: bool) -> str:
    return "true" if value else "false"


def run_benchmark(args) -> Path:
    output_root = Path(args.benchmark_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in output_root.iterdir() if path.is_dir()}

    command = [
        sys.executable,
        "benchmark_models.py",
        "--datasets",
        ",".join(args.datasets),
        "--models",
        ",".join(args.models),
        "--smd_groups",
        ",".join(args.smd_groups),
        "--lookback",
        str(args.lookback),
        "--normalize",
        bool_to_cli(args.normalize),
        "--batch_size",
        str(args.batch_size),
        "--val_split",
        str(args.val_split),
        "--weight_decay",
        str(args.weight_decay),
        "--gamma",
        str(args.gamma),
        "--seed",
        str(args.seed),
        "--use_cuda",
        bool_to_cli(args.use_cuda),
        "--use_gatv2",
        bool_to_cli(args.use_gatv2),
        "--output_root",
        str(output_root),
        "--max_train_size",
        str(args.max_train_size),
        "--max_test_size",
        str(args.max_test_size),
        "--mtad_epochs",
        str(args.mtad_epochs),
        "--mtad_lr",
        str(args.mtad_lr),
        "--mtad_gru_hid_dim",
        str(args.mtad_gru_hid_dim),
        "--mtad_fc_hid_dim",
        str(args.mtad_fc_hid_dim),
        "--mtad_recon_hid_dim",
        str(args.mtad_recon_hid_dim),
        "--mtad_node_embed_dim",
        str(args.mtad_node_embed_dim),
        "--mtad_vae_latent_dim",
        str(args.mtad_vae_latent_dim),
        "--mtad_kl_beta",
        str(args.mtad_kl_beta),
    ]

    print("开始运行主模型筛选 benchmark：")
    print(" ".join(command))
    subprocess.run(command, check=True)

    after = [path.resolve() for path in output_root.iterdir() if path.is_dir()]
    new_dirs = sorted([path for path in after if path not in before], key=lambda item: item.stat().st_mtime)
    if new_dirs:
        return new_dirs[-1]
    return max(after, key=lambda item: item.stat().st_mtime)


def summarize_run(run_dir: Path, selection_root: Path, primary_selector: str) -> Path:
    result_csv = run_dir / "benchmark_results.csv"
    if not result_csv.exists():
        raise FileNotFoundError(f"未找到结果文件：{result_csv}")

    df = pd.read_csv(result_csv)
    per_entity = (
        df.pivot_table(
            index=["dataset", "entity", "model"],
            columns="threshold_method",
            values="f1",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "epsilon_result": "epsilon_f1",
                "pot_result": "pot_f1",
                "bf_result": "bf_f1",
            }
        )
    )

    leaderboard = (
        per_entity.groupby("model", as_index=False)[["epsilon_f1", "pot_f1", "bf_f1"]]
        .mean()
        .sort_values(["bf_f1", "pot_f1", "epsilon_f1"], ascending=False)
        .reset_index(drop=True)
    )

    for metric in ["epsilon_f1", "pot_f1", "bf_f1"]:
        leaderboard[f"{metric}_rank"] = leaderboard[metric].rank(ascending=False, method="min")

    leaderboard["mean_f1"] = leaderboard[["epsilon_f1", "pot_f1", "bf_f1"]].mean(axis=1)
    leaderboard["mean_rank"] = leaderboard[["epsilon_f1_rank", "pot_f1_rank", "bf_f1_rank"]].mean(axis=1)

    if primary_selector == "mean_rank":
        leaderboard = leaderboard.sort_values(
            ["mean_rank", "mean_f1", "bf_f1", "pot_f1", "epsilon_f1"],
            ascending=[True, False, False, False, False],
        )
    else:
        leaderboard = leaderboard.sort_values(
            [primary_selector, "mean_rank", "mean_f1"],
            ascending=[False, True, False],
        )

    recommended_model = str(leaderboard.iloc[0]["model"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = selection_root / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)

    per_entity.to_csv(summary_dir / "per_entity_f1.csv", index=False)
    leaderboard.to_csv(summary_dir / "leaderboard.csv", index=False)

    with open(summary_dir / "recommended_model.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_dir": str(run_dir),
                "primary_selector": primary_selector,
                "recommended_model": recommended_model,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    md_table = dataframe_to_markdown(leaderboard)
    markdown_lines = [
        "# 主模型筛选结果",
        "",
        f"- benchmark 结果目录：`{run_dir}`",
        f"- 排序规则：`{primary_selector}`",
        f"- 推荐主模型：`{recommended_model}`",
        "",
        "## 模型排行榜",
        "",
        md_table,
        "",
    ]
    (summary_dir / "leaderboard.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    print(f"\n推荐主模型：{recommended_model}")
    print(f"筛选结果已保存到：{summary_dir}")
    return summary_dir


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [headers, ["---"] * len(headers)]
    for _, row in df.iterrows():
        rows.append([str(row[col]) for col in headers])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = run_benchmark(args)

    summarize_run(
        run_dir=run_dir,
        selection_root=Path(args.selection_root),
        primary_selector=args.primary_selector,
    )


if __name__ == "__main__":
    main()
