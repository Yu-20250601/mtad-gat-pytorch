import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, Dataset, random_split
from eval_methods import bf_search, epsilon_eval, pot_eval
from mtad_gat import MTAD_GAT
from utils import adjust_anomaly_scores, get_data


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("布尔参数只接受 true/false、yes/no、1/0 等形式。")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_parent(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_default_pot_params(dataset: str, entity: Optional[str]) -> Tuple[float, float, int]:
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
    }
    reg_level_dict = {
        "SMAP": 0,
        "MSL": 0,
        "SMD-1": 1,
        "SMD-2": 1,
        "SMD-3": 1,
    }
    if dataset == "SMD":
        key = f"SMD-{entity.split('-')[0]}"
    else:
        key = dataset
    level, q = level_q_dict[key]
    reg_level = reg_level_dict[key]
    return level, q, reg_level


def infer_smd_entities() -> List[str]:
    processed_dir = Path("datasets/ServerMachineDataset/processed")
    if not processed_dir.exists():
        return []
    entities = []
    for file_path in sorted(processed_dir.glob("machine-*_train.pkl")):
        name = file_path.name.replace("_train.pkl", "")
        parts = name.split("-")
        entities.append(f"{parts[1]}-{parts[2]}")
    return entities


def validate_dataset_ready(dataset: str) -> None:
    if dataset == "SMD":
        processed_dir = Path("datasets/ServerMachineDataset/processed")
        if not processed_dir.exists():
            raise FileNotFoundError(
                "未找到 SMD 的预处理文件，请先执行 `python preprocess.py --dataset SMD`。"
            )
        return

    expected = Path(f"datasets/data/processed/{dataset}_train.pkl")
    if not expected.exists():
        raise FileNotFoundError(
            f"未找到 {dataset} 的预处理文件：{expected}。"
            f"请先下载原始遥测数据，再执行 `python preprocess.py --dataset {dataset}`。"
        )


def load_entity_data(
    dataset: str,
    entity: Optional[str],
    normalize: bool,
    max_train_size: Optional[int] = None,
    max_test_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "SMD":
        machine_name = f"machine-{entity}"
        (x_train, _), (x_test, y_test) = get_data(
            machine_name,
            normalize=normalize,
            max_train_size=max_train_size,
            max_test_size=max_test_size,
        )
    else:
        (x_train, _), (x_test, y_test) = get_data(
            dataset,
            normalize=normalize,
            max_train_size=max_train_size,
            max_test_size=max_test_size,
        )
    return x_train.astype(np.float32), x_test.astype(np.float32), y_test.astype(np.int64)


class WindowForecastDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int):
        self.data = torch.from_numpy(data).float()
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.data) - self.window_size

    def __getitem__(self, index: int):
        x = self.data[index : index + self.window_size]
        y = self.data[index + self.window_size]
        return x, y


class GDNBaseline(nn.Module):
    """一个受 GDN 启发的轻量图预测基线模型。"""

    def __init__(self, n_features: int, window_size: int, hidden_dim: int, topk: int, dropout: float):
        super().__init__()
        self.n_features = n_features
        self.topk = min(max(1, topk), n_features)
        self.node_embedding = nn.Embedding(n_features, hidden_dim)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.message_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _adjacency(self) -> torch.Tensor:
        emb = torch.nn.functional.normalize(self.node_embedding.weight, p=2, dim=1)
        sim = emb @ emb.T
        if self.topk < self.n_features:
            _, idx = torch.topk(sim, self.topk, dim=1)
            mask = torch.full_like(sim, float("-inf"))
            mask.scatter_(1, idx, sim.gather(1, idx))
            sim = mask
        return torch.softmax(sim, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入张量形状为 (batch, window, features)
        x = x.permute(0, 2, 1)
        node_repr = self.temporal_encoder(x)
        adjacency = self._adjacency()
        propagated = torch.einsum("ij,bjh->bih", adjacency, self.message_proj(node_repr))
        fused = torch.cat([node_repr, propagated], dim=-1)
        out = self.out_proj(fused).squeeze(-1)
        return out


@dataclass
class ScoreOutput:
    train_scores: np.ndarray
    test_scores: np.ndarray


def make_window_matrix(data: np.ndarray, window_size: int) -> np.ndarray:
    windows = [data[i : i + window_size] for i in range(len(data) - window_size)]
    return np.stack(windows, axis=0)


def score_pca(x_train: np.ndarray, x_test: np.ndarray, window_size: int, variance_ratio: float) -> ScoreOutput:
    train_windows = make_window_matrix(x_train, window_size).reshape(len(x_train) - window_size, -1)
    test_windows = make_window_matrix(x_test, window_size).reshape(len(x_test) - window_size, -1)
    pca = PCA(n_components=variance_ratio, svd_solver="full")
    pca.fit(train_windows)

    def reconstruction_error(matrix: np.ndarray) -> np.ndarray:
        recon = pca.inverse_transform(pca.transform(matrix))
        error = (matrix - recon).reshape(matrix.shape[0], window_size, x_train.shape[1])
        return np.mean(np.square(error[:, -1, :]), axis=1)

    return ScoreOutput(reconstruction_error(train_windows), reconstruction_error(test_windows))


def score_isolation_forest(
    x_train: np.ndarray,
    x_test: np.ndarray,
    window_size: int,
    n_estimators: int,
    contamination: str,
    random_state: int,
) -> ScoreOutput:
    train_windows = make_window_matrix(x_train, window_size).reshape(len(x_train) - window_size, -1)
    test_windows = make_window_matrix(x_test, window_size).reshape(len(x_test) - window_size, -1)
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    clf.fit(train_windows)
    train_scores = -clf.score_samples(train_windows)
    test_scores = -clf.score_samples(test_windows)
    return ScoreOutput(train_scores, test_scores)


def train_forecast_model(
    model: nn.Module,
    train_data: np.ndarray,
    window_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_split: float,
    device: torch.device,
) -> None:
    dataset = WindowForecastDataset(train_data, window_size)
    if len(dataset) == 0:
        raise ValueError("训练集长度小于 lookback，无法构造滑动窗口样本。")

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_subset, val_subset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        train_subset = dataset
        val_loader = None

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    model.to(device)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        if val_loader is None:
            continue

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                val_losses.append(criterion(pred, batch_y).item())
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)


def collect_forecast_errors(
    model: nn.Module,
    data: np.ndarray,
    window_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = WindowForecastDataset(data, window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            err = torch.mean(torch.square(pred - batch_y), dim=1)
            scores.append(err.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def score_gdn(
    x_train: np.ndarray,
    x_test: np.ndarray,
    window_size: int,
    hidden_dim: int,
    topk: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_split: float,
    device: torch.device,
) -> ScoreOutput:
    model = GDNBaseline(
        n_features=x_train.shape[1],
        window_size=window_size,
        hidden_dim=hidden_dim,
        topk=topk,
        dropout=0.1,
    )
    train_forecast_model(
        model,
        x_train,
        window_size=window_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        val_split=val_split,
        device=device,
    )
    return ScoreOutput(
        collect_forecast_errors(model, x_train, window_size, batch_size, device),
        collect_forecast_errors(model, x_test, window_size, batch_size, device),
    )


def train_mtad_model(
    x_train: np.ndarray,
    window_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_split: float,
    device: torch.device,
    use_sparsemax: bool,
    use_gatv2: bool,
    recon_model: str,
    use_node_embedding: bool,
    gru_hid_dim: int,
    fc_hid_dim: int,
    recon_hid_dim: int,
    node_embed_dim: int,
    vae_latent_dim: int,
    kl_beta: float,
) -> MTAD_GAT:
    dataset = WindowForecastDataset(x_train, window_size)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_subset, val_subset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        train_subset = dataset
        val_loader = None

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    n_features = x_train.shape[1]
    model = MTAD_GAT(
        n_features=n_features,
        window_size=window_size,
        out_dim=n_features,
        kernel_size=7,
        use_gatv2=use_gatv2,
        gru_n_layers=1,
        gru_hid_dim=gru_hid_dim,
        forecast_n_layers=3,
        forecast_hid_dim=fc_hid_dim,
        recon_n_layers=1,
        recon_hid_dim=recon_hid_dim,
        dropout=0.3,
        alpha=0.2,
        node_embed_dim=node_embed_dim,
        use_node_embedding=use_node_embedding,
        use_sparsemax=use_sparsemax,
        recon_model=recon_model,
        vae_latent_dim=vae_latent_dim,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            model_out = model(batch_x)
            if isinstance(model_out, (tuple, list)) and len(model_out) == 3:
                preds, recons, kl = model_out
            else:
                preds, recons = model_out
                kl = None
            recon_target = batch_x
            loss = torch.sqrt(criterion(preds, batch_y)) + torch.sqrt(criterion(recons, recon_target))
            if kl is not None:
                loss = loss + kl_beta * kl
            loss.backward()
            optimizer.step()

        if val_loader is None:
            continue

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                model_out = model(batch_x)
                if isinstance(model_out, (tuple, list)) and len(model_out) == 3:
                    preds, recons, kl = model_out
                else:
                    preds, recons = model_out
                    kl = None
                loss = torch.sqrt(criterion(preds, batch_y)) + torch.sqrt(criterion(recons, batch_x))
                if kl is not None:
                    loss = loss + kl_beta * kl
                val_losses.append(loss.item())
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def collect_mtad_scores(
    model: MTAD_GAT,
    data: np.ndarray,
    window_size: int,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> np.ndarray:
    tensor_data = torch.from_numpy(data).float()
    dataset = WindowForecastDataset(data, window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds_list = []
    recons_list = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            model_out = model(batch_x)
            if isinstance(model_out, (tuple, list)) and len(model_out) == 3:
                preds, _recons, _kl = model_out
            else:
                preds, _recons = model_out
            recon_input = torch.cat((batch_x[:, 1:, :], batch_y.unsqueeze(1)), dim=1)
            recon_out = model(recon_input)
            if isinstance(recon_out, (tuple, list)) and len(recon_out) == 3:
                _preds, recons, _kl = recon_out
            else:
                _preds, recons = recon_out
            preds_list.append(preds.detach().cpu().numpy())
            recons_list.append(recons[:, -1, :].detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    recons = np.concatenate(recons_list, axis=0)
    actual = tensor_data[window_size:].numpy()
    forecast_error = np.sqrt(np.square(preds - actual))
    recon_error = np.sqrt(np.square(recons - actual))
    scores = np.mean(forecast_error + gamma * recon_error, axis=1)
    return scores


def score_mtad(
    x_train: np.ndarray,
    x_test: np.ndarray,
    dataset: str,
    entity: Optional[str],
    window_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_split: float,
    gamma: float,
    device: torch.device,
    use_sparsemax: bool,
    use_gatv2: bool,
    recon_model: str,
    use_node_embedding: bool,
    gru_hid_dim: int,
    fc_hid_dim: int,
    recon_hid_dim: int,
    node_embed_dim: int,
    vae_latent_dim: int,
    kl_beta: float,
) -> ScoreOutput:
    model = train_mtad_model(
        x_train=x_train,
        window_size=window_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        val_split=val_split,
        device=device,
        use_sparsemax=use_sparsemax,
        use_gatv2=use_gatv2,
        recon_model=recon_model,
        use_node_embedding=use_node_embedding,
        gru_hid_dim=gru_hid_dim,
        fc_hid_dim=fc_hid_dim,
        recon_hid_dim=recon_hid_dim,
        node_embed_dim=node_embed_dim,
        vae_latent_dim=vae_latent_dim,
        kl_beta=kl_beta,
    )
    train_scores = collect_mtad_scores(model, x_train, window_size, batch_size, gamma, device)
    test_scores = collect_mtad_scores(model, x_test, window_size, batch_size, gamma, device)
    train_scores = adjust_anomaly_scores(train_scores, dataset, True, window_size)
    test_scores = adjust_anomaly_scores(test_scores, dataset, False, window_size)
    return ScoreOutput(train_scores, test_scores)


def evaluate_scores(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
    dataset: str,
    entity: Optional[str],
) -> Dict[str, Dict[str, float]]:
    level, q, reg_level = get_default_pot_params(dataset, entity)
    epsilon_result = epsilon_eval(train_scores, test_scores, labels, reg_level=reg_level)
    pot_result = pot_eval(train_scores, test_scores, labels, q=q, level=level, dynamic=False)
    bf_result = bf_search(test_scores, labels, start=0.01, end=2.0, step_num=100, verbose=False)

    for result in (epsilon_result, pot_result, bf_result):
        for key, value in list(result.items()):
            if isinstance(value, (np.floating, np.integer)):
                result[key] = float(value)
    return {
        "epsilon_result": epsilon_result,
        "pot_result": pot_result,
        "bf_result": bf_result,
    }


def resolve_mtad_variant(model_name: str, args) -> Optional[Dict[str, object]]:
    alias_map = {
        "mtad_gat": "mtad_gat_gru",
        "mtad_gat_sparsemax": "mtad_gat_sparsemax_gru",
    }
    canonical_name = alias_map.get(model_name, model_name)

    variant_map = {
        "mtad_gat_gru": {"recon_model": "gru", "use_sparsemax": False, "use_node_embedding": False},
        "mtad_gat_gru_node": {"recon_model": "gru", "use_sparsemax": False, "use_node_embedding": True},
        "mtad_gat_sparsemax_gru": {"recon_model": "gru", "use_sparsemax": True, "use_node_embedding": False},
        "mtad_gat_sparsemax_gru_node": {"recon_model": "gru", "use_sparsemax": True, "use_node_embedding": True},
        "mtad_gat_vae": {"recon_model": "vae", "use_sparsemax": False, "use_node_embedding": False},
        "mtad_gat_vae_node": {"recon_model": "vae", "use_sparsemax": False, "use_node_embedding": True},
        "mtad_gat_sparsemax_vae": {"recon_model": "vae", "use_sparsemax": True, "use_node_embedding": False},
        "mtad_gat_sparsemax_vae_node": {"recon_model": "vae", "use_sparsemax": True, "use_node_embedding": True},
        "mtad_gat_custom": {
            "recon_model": args.mtad_custom_recon_model,
            "use_sparsemax": args.mtad_use_sparsemax,
            "use_node_embedding": args.mtad_use_node_embedding,
        },
    }

    if canonical_name not in variant_map:
        return None

    resolved = dict(variant_map[canonical_name])
    resolved["canonical_name"] = canonical_name
    return resolved


def run_single_model(
    model_name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    labels: np.ndarray,
    dataset: str,
    entity: Optional[str],
    args,
    device: torch.device,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    if model_name == "pca":
        scores = score_pca(x_train, x_test, args.lookback, args.pca_variance_ratio)
        resolved_name = "pca"
    elif model_name == "isolation_forest":
        scores = score_isolation_forest(
            x_train,
            x_test,
            args.lookback,
            args.iforest_estimators,
            args.iforest_contamination,
            args.seed,
        )
        resolved_name = "isolation_forest"
    elif model_name == "gdn":
        scores = score_gdn(
            x_train,
            x_test,
            args.lookback,
            args.gdn_hidden_dim,
            args.gdn_topk,
            args.batch_size,
            args.gdn_epochs,
            args.gdn_lr,
            args.weight_decay,
            args.val_split,
            device,
        )
        resolved_name = "gdn"
    else:
        mtad_variant = resolve_mtad_variant(model_name, args)
        if mtad_variant is None:
            raise ValueError(f"暂不支持的模型名称: {model_name}")

        resolved_name = mtad_variant["canonical_name"]
        scores = score_mtad(
            x_train,
            x_test,
            dataset,
            entity,
            args.lookback,
            args.batch_size,
            args.mtad_epochs,
            args.mtad_lr,
            args.weight_decay,
            args.val_split,
            args.gamma,
            device,
            use_sparsemax=bool(mtad_variant["use_sparsemax"]),
            use_gatv2=args.use_gatv2,
            recon_model=str(mtad_variant["recon_model"]),
            use_node_embedding=bool(mtad_variant["use_node_embedding"]),
            gru_hid_dim=args.mtad_gru_hid_dim,
            fc_hid_dim=args.mtad_fc_hid_dim,
            recon_hid_dim=args.mtad_recon_hid_dim,
            node_embed_dim=args.mtad_node_embed_dim,
            vae_latent_dim=args.mtad_vae_latent_dim,
            kl_beta=args.mtad_kl_beta,
        )

    return resolved_name, evaluate_scores(scores.train_scores, scores.test_scores, labels, dataset, entity)


def flatten_results(
    dataset: str,
    entity: Optional[str],
    model_name: str,
    metrics: Dict[str, Dict[str, float]],
) -> List[Dict[str, object]]:
    rows = []
    for method_name, method_metrics in metrics.items():
        row = {
            "dataset": dataset,
            "entity": entity or dataset,
            "model": model_name,
            "threshold_method": method_name,
        }
        row.update(method_metrics)
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一异常检测对比实验脚本。")
    parser.add_argument("--datasets", type=parse_list_arg, default=["SMD", "MSL", "SMAP"])
    parser.add_argument(
        "--models",
        type=parse_list_arg,
        default=[
            "pca",
            "isolation_forest",
            "mtad_gat_gru",
            "mtad_gat_vae",
            "mtad_gat_sparsemax_vae",
        ],
    )
    parser.add_argument("--smd_groups", type=parse_list_arg, default=["all"])
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--output_root", type=str, default="benchmark_output")
    parser.add_argument("--max_train_size", type=int, default=None)
    parser.add_argument("--max_test_size", type=int, default=None)

    parser.add_argument("--pca_variance_ratio", type=float, default=0.95)
    parser.add_argument("--iforest_estimators", type=int, default=200)
    parser.add_argument("--iforest_contamination", type=str, default="auto")

    parser.add_argument("--gdn_epochs", type=int, default=20)
    parser.add_argument("--gdn_lr", type=float, default=1e-3)
    parser.add_argument("--gdn_hidden_dim", type=int, default=64)
    parser.add_argument("--gdn_topk", type=int, default=10)

    parser.add_argument("--mtad_epochs", type=int, default=30)
    parser.add_argument("--mtad_lr", type=float, default=1e-3)
    parser.add_argument("--mtad_gru_hid_dim", type=int, default=150)
    parser.add_argument("--mtad_fc_hid_dim", type=int, default=150)
    parser.add_argument("--mtad_recon_hid_dim", type=int, default=150)
    parser.add_argument("--mtad_node_embed_dim", type=int, default=16)
    parser.add_argument("--mtad_vae_latent_dim", type=int, default=64)
    parser.add_argument("--mtad_kl_beta", type=float, default=0.001)
    parser.add_argument("--mtad_use_sparsemax", type=str2bool, default=False)
    parser.add_argument("--mtad_use_node_embedding", type=str2bool, default=False)
    parser.add_argument("--mtad_custom_recon_model", type=str.lower, default="gru", choices=["gru", "vae"])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / timestamp
    ensure_parent(run_dir)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    all_rows = []
    for dataset in args.datasets:
        dataset = dataset.upper()
        validate_dataset_ready(dataset)

        if dataset == "SMD":
            entities = infer_smd_entities() if args.smd_groups == ["all"] else args.smd_groups
        else:
            entities = [None]

        for entity in entities:
            entity_tag = entity or dataset
            print(f"\n=== 数据集={dataset} 实体={entity_tag} ===")
            x_train, x_test, y_test = load_entity_data(
                dataset,
                entity,
                normalize=args.normalize,
                max_train_size=args.max_train_size,
                max_test_size=args.max_test_size,
            )
            labels = y_test[args.lookback:]

            for model_name in args.models:
                print(f"正在运行 {model_name} ...")
                resolved_name, metrics = run_single_model(model_name, x_train, x_test, labels, dataset, entity, args, device)
                model_dir = run_dir / dataset / entity_tag / resolved_name
                ensure_parent(model_dir)
                with open(model_dir / "metrics.json", "w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, indent=2)
                all_rows.extend(flatten_results(dataset, entity, resolved_name, metrics))

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(run_dir / "benchmark_results.csv", index=False)

    summary_df = (
        result_df.sort_values(["dataset", "entity", "threshold_method", "f1"], ascending=[True, True, True, False])
        .groupby(["dataset", "entity", "threshold_method"], as_index=False)
        .first()
    )
    summary_df.to_csv(run_dir / "best_per_entity.csv", index=False)

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    print(f"\n对比实验结果已保存到 {run_dir}")


if __name__ == "__main__":
    main()
