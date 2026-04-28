import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def bool_to_cli(value: bool) -> str:
    return "true" if value else "false"


def run_command(command: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def latest_run_dir(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def model_to_train_args(model_name: str) -> List[str]:
    mapping = {
        "mtad_gat_gru": [
            "--use_adaptive_sparse_feat_gat",
            "false",
            "--use_node_embedding",
            "false",
            "--use_sparsemax",
            "false",
            "--recon_model",
            "gru",
        ],
        "mtad_gat_vae": [
            "--use_adaptive_sparse_feat_gat",
            "false",
            "--use_node_embedding",
            "false",
            "--use_sparsemax",
            "false",
            "--recon_model",
            "vae",
            "--vae_latent_dim",
            "64",
            "--kl_beta",
            "0.001",
        ],
        "mtad_gat_sparsemax_vae": [
            "--use_adaptive_sparse_feat_gat",
            "true",
            "--use_node_embedding",
            "false",
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
        ],
    }
    if model_name not in mapping:
        raise KeyError(f"Unsupported thesis model: {model_name}")
    return mapping[model_name]


def run_benchmark(
    repo_root: Path,
    output_root: Path,
    groups: List[str],
    models: List[str],
    seed: int,
    use_gatv2: bool,
    run_name: str,
    epochs: int,
    batch_size: int,
) -> Path:
    benchmark_root = output_root / run_name
    command = [
        sys.executable,
        "benchmark_models.py",
        "--datasets",
        "SMD",
        "--smd_groups",
        ",".join(groups),
        "--models",
        ",".join(models),
        "--mtad_epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--use_gatv2",
        bool_to_cli(use_gatv2),
        "--seed",
        str(seed),
        "--output_root",
        str(benchmark_root),
    ]
    run_command(command, repo_root, output_root / "logs" / f"{run_name}.log")
    return latest_run_dir(benchmark_root)


def run_detailed_train(
    repo_root: Path,
    output_root: Path,
    group: str,
    model_name: str,
    use_gatv2: bool,
    epochs: int,
    batch_size: int,
) -> Path:
    group_root = repo_root / "output" / "SMD" / group
    before = {path.resolve() for path in group_root.iterdir() if path.is_dir()} if group_root.exists() else set()

    command = [
        sys.executable,
        "train.py",
        "--dataset",
        "SMD",
        "--group",
        group,
        "--epochs",
        str(epochs),
        "--lookback",
        "100",
        "--bs",
        str(batch_size),
        "--init_lr",
        "1e-3",
        "--dropout",
        "0.3",
        "--val_split",
        "0.1",
        "--use_gatv2",
        bool_to_cli(use_gatv2),
    ] + model_to_train_args(model_name)

    safe_name = f"detail_{group}_{model_name}_{'gatv2' if use_gatv2 else 'gat'}"
    run_command(command, repo_root, output_root / "logs" / f"{safe_name}.log")

    after = [path.resolve() for path in group_root.iterdir() if path.is_dir()]
    new_dirs = [path for path in after if path not in before and path.name != "logs"]
    if new_dirs:
        return max(new_dirs, key=lambda item: item.stat().st_mtime)
    candidates = [path for path in after if path.name != "logs"]
    if not candidates:
        raise FileNotFoundError(f"No detailed run directory found for group {group}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def analyze_run(repo_root: Path, output_root: Path, group: str, run_dir: Path) -> None:
    command = [
        sys.executable,
        "analyze_results.py",
        "--dataset",
        "SMD",
        "--group",
        group,
        "--target_dir",
        str(run_dir),
    ]
    log_name = f"analyze_{group}_{run_dir.name}.log"
    run_command(command, repo_root, output_root / "logs" / log_name)


def render_case_figure(
    repo_root: Path,
    output_root: Path,
    run_dirs: List[Path],
    labels: List[str],
    title: str,
    output_file: str,
) -> None:
    command = [
        sys.executable,
        "render_case_figures.py",
        "--run_dirs",
    ] + [str(item) for item in run_dirs] + [
        "--labels",
        ",".join(labels),
        "--threshold_method",
        "pot_result",
        "--output",
        str(output_root / "paper_assets" / "figures" / output_file),
        "--title",
        title,
        "--context",
        "250",
    ]
    run_command(command, repo_root, output_root / "logs" / f"figure_{output_file}.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full thesis experiment pipeline end to end.")
    parser.add_argument("--groups", type=parse_list_arg, default=["1-2", "2-1", "3-2"])
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
    parser.add_argument("--case_groups", type=parse_list_arg, default=["1-2", "3-2"])
    parser.add_argument("--main_seed", type=int, default=42)
    parser.add_argument("--stability_seeds", type=parse_list_arg, default=["42", "52", "62"])
    parser.add_argument("--benchmark_epochs", type=int, default=15)
    parser.add_argument("--detail_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_root", type=str, default="thesis_artifacts")
    parser.add_argument("--proposed_model", type=str, default="mtad_gat_sparsemax_vae_node")
    parser.add_argument("--baseline_model", type=str, default="mtad_gat_gru")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "paper_assets" / "figures").mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "groups": args.groups,
        "models": args.models,
        "case_groups": args.case_groups,
    }

    main_run = run_benchmark(
        repo_root,
        output_root,
        args.groups,
        args.models,
        args.main_seed,
        True,
        "main_benchmark",
        args.benchmark_epochs,
        args.batch_size,
    )
    manifest["main_run"] = str(main_run)

    stability_runs: List[Path] = []
    for seed_text in args.stability_seeds:
        seed = int(seed_text)
        if seed == args.main_seed:
            stability_runs.append(main_run)
            continue
        run_dir = run_benchmark(
            repo_root,
            output_root,
            args.groups,
            args.models,
            seed,
            True,
            f"stability_seed_{seed}",
            args.benchmark_epochs,
            args.batch_size,
        )
        stability_runs.append(run_dir)
    manifest["stability_runs"] = [str(item) for item in stability_runs]

    gatv2_true_run = run_benchmark(
        repo_root,
        output_root,
        args.groups,
        [args.proposed_model],
        args.main_seed,
        True,
        "gatv2_true",
        args.benchmark_epochs,
        args.batch_size,
    )
    gatv2_false_run = run_benchmark(
        repo_root,
        output_root,
        args.groups,
        [args.proposed_model],
        args.main_seed,
        False,
        "gatv2_false",
        args.benchmark_epochs,
        args.batch_size,
    )
    manifest["gatv2_true_run"] = str(gatv2_true_run)
    manifest["gatv2_false_run"] = str(gatv2_false_run)

    detailed_runs: Dict[str, Dict[str, str]] = {}
    for group in args.case_groups:
        baseline_run = run_detailed_train(
            repo_root,
            output_root,
            group,
            args.baseline_model,
            True,
            args.detail_epochs,
            args.batch_size,
        )
        analyze_run(repo_root, output_root, group, baseline_run)

        proposed_run = run_detailed_train(
            repo_root,
            output_root,
            group,
            args.proposed_model,
            True,
            args.detail_epochs,
            args.batch_size,
        )
        analyze_run(repo_root, output_root, group, proposed_run)

        detailed_runs[group] = {
            args.baseline_model: str(baseline_run),
            args.proposed_model: str(proposed_run),
        }

    manifest["detailed_runs"] = detailed_runs

    for group in args.case_groups:
        render_case_figure(
            repo_root,
            output_root,
            [
                Path(detailed_runs[group][args.baseline_model]),
                Path(detailed_runs[group][args.proposed_model]),
            ],
            ["Baseline-GRU", "Proposed-SparseVAE+Node"],
            f"SMD {group} case comparison (POT)",
            f"case_compare_{group}_pot.png",
        )

    aggregate_command = [
        sys.executable,
        "aggregate_thesis_results.py",
        "--main_run",
        str(main_run),
        "--output_dir",
        str(output_root / "paper_assets"),
        "--seed_runs",
        ",".join(str(item) for item in stability_runs),
        "--gatv2_true_run",
        str(gatv2_true_run),
        "--gatv2_false_run",
        str(gatv2_false_run),
    ]
    run_command(aggregate_command, repo_root, output_root / "logs" / "aggregate.log")

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Thesis pipeline completed. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
