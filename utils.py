import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class LoadedDataset:
    name: str
    path: Path
    edge_indices: Sequence[np.ndarray]
    edge_weights: Sequence[np.ndarray]
    feats: np.ndarray
    labels: np.ndarray
    mask: np.ndarray


@dataclass
class Snapshot:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    y: np.ndarray


@dataclass
class GroupStats:
    name: str
    per_snapshot_acc: List[float]
    per_snapshot_f1: List[float]
    per_snapshot_count: List[int]
    pooled_y_true: List[np.ndarray]
    pooled_y_pred: List[np.ndarray]

    def support(self) -> int:
        return int(np.sum(self.per_snapshot_count))

    def mean_acc(self) -> float:
        return nanmean(self.per_snapshot_acc)

    def mean_f1(self) -> float:
        return nanmean(self.per_snapshot_f1)

    def pooled_acc(self) -> float:
        return compute_acc(safe_concat(self.pooled_y_true), safe_concat(self.pooled_y_pred))

    def pooled_f1(self, average: str) -> float:
        return compute_f1(safe_concat(self.pooled_y_true), safe_concat(self.pooled_y_pred), average=average)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float:
    if y_true.size == 0:
        return np.nan
    try:
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    except ValueError:
        return np.nan


def compute_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return np.nan
    return float(accuracy_score(y_true, y_pred))


def nanmean(xs: List[float]) -> float:
    arr = np.array(xs, dtype=float)
    return float(np.nanmean(arr)) if arr.size > 0 else np.nan


def safe_concat(parts: List[np.ndarray]) -> np.ndarray:
    if len(parts) == 0:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts).astype(np.int64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensors(snapshot: Snapshot, device: torch.device):
    x = torch.tensor(snapshot.x, dtype=torch.float32, device=device)
    edge_index = torch.tensor(snapshot.edge_index, dtype=torch.long, device=device)
    edge_attr = torch.tensor(snapshot.edge_attr, dtype=torch.float32, device=device)
    y = torch.tensor(snapshot.y, dtype=torch.float32, device=device)
    return x, edge_index, edge_attr, y


def list_dataset_files(data_dir: Path) -> Dict[str, Path]:
    dataset_map: Dict[str, Path] = {}
    if not data_dir.exists():
        return dataset_map
    for p in sorted(data_dir.glob("*.pkl")):
        dataset_map[p.stem.lower()] = p
    return dataset_map


def resolve_dataset_path(dataset_name_or_path: str, data_dir: Path) -> Path:
    candidate = Path(dataset_name_or_path)
    if candidate.exists() and candidate.is_file():
        return candidate

    dataset_map = list_dataset_files(data_dir)
    key = dataset_name_or_path.lower()
    if key in dataset_map:
        return dataset_map[key]
    raise FileNotFoundError(
        f"Dataset '{dataset_name_or_path}' not found. Available dataset keys: {sorted(dataset_map.keys())}"
    )


def load_dataset(dataset_name_or_path: str, data_dir: Path) -> LoadedDataset:
    path = resolve_dataset_path(dataset_name_or_path, data_dir)
    with path.open("rb") as f:
        data = pickle.load(f)

    required = ["edge_indices", "edge_weights", "feats", "labels", "mask"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Dataset file {path} missing keys: {missing}")

    edge_indices = data["edge_indices"]
    edge_weights = data["edge_weights"]
    feats = np.asarray(data["feats"])
    labels = np.asarray(data["labels"])
    mask = np.asarray(data["mask"])

    if labels.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"Expected labels/mask with shape [K, N], got labels={labels.shape}, mask={mask.shape}")
    if feats.ndim != 3:
        raise ValueError(f"Expected feats with shape [K, N, F], got feats={feats.shape}")
    if not (len(edge_indices) == len(edge_weights) == feats.shape[0] == labels.shape[0] == mask.shape[0]):
        raise ValueError("Snapshot length mismatch among edge_indices/edge_weights/feats/labels/mask")

    return LoadedDataset(
        name=path.stem,
        path=path,
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        feats=feats,
        labels=labels,
        mask=mask,
    )


def build_temporal_dataset(data: LoadedDataset) -> List[Snapshot]:
    out: List[Snapshot] = []
    for t in range(data.labels.shape[0]):
        out.append(
            Snapshot(
                x=data.feats[t],
                edge_index=data.edge_indices[t],
                edge_attr=data.edge_weights[t],
                y=data.labels[t].astype(np.float32),
            )
        )
    return out


def resolve_split(dataset_name: str, K: int, train_T: Optional[int], val_T: Optional[int]) -> Tuple[int, int, int]:
    """Returns (train_T, val_T, test_T); falls back to ~60/20/20 if no override matches."""
    if train_T is None or val_T is None:
        name = dataset_name.lower()

        if name.startswith("dblp") and K >= 12:
            tr, va = 9, 2
            te = K - tr - va
            if te >= 1:
                return tr, va, te

        elif name.startswith("gossipcop") and K >= 32:
            tr, va = 26, 5
            te = K - tr - va
            if te >= 1:
                return tr, va, te

        elif name.startswith("music") and K >= 20:
            tr, va = 15, 4
            te = K - tr - va
            if te >= 1:
                return tr, va, te

        tr = max(1, int(round(0.6 * K)))
        va = max(1, int(round(0.2 * K)))
        while tr + va >= K:
            if va > 1:
                va -= 1
            else:
                tr = max(1, tr - 1)
        te = K - tr - va
        return tr, va, te

    if train_T < 1 or val_T < 1:
        raise ValueError("train_T and val_T must be >= 1")
    te = K - train_T - val_T
    if te < 1:
        raise ValueError(f"Invalid split: K={K}, train_T={train_T}, val_T={val_T} leaves no test snapshots")
    return train_T, val_T, te


def evaluate_test_groups(
    labels: np.ndarray,
    evolving_mask: np.ndarray,
    snapshot_preds: List[np.ndarray],
    test_start_t: int,
    f1_average: str,
) -> Dict[str, GroupStats]:
    groups = {
        "all":       GroupStats("all",       [], [], [], [], []),
        "changed":   GroupStats("changed",   [], [], [], [], []),
        "unchanged": GroupStats("unchanged", [], [], [], [], []),
    }

    for local_i, pred in enumerate(snapshot_preds):
        t = test_start_t + local_i
        m_t = evolving_mask[t].astype(bool)

        all_true = labels[t][m_t]
        all_pred = pred[m_t]
        groups["all"].per_snapshot_count.append(len(all_true))
        groups["all"].per_snapshot_acc.append(compute_acc(all_true, all_pred))
        groups["all"].per_snapshot_f1.append(compute_f1(all_true, all_pred, average=f1_average))
        groups["all"].pooled_y_true.append(all_true)
        groups["all"].pooled_y_pred.append(all_pred)

        if t - 1 >= 0:
            both_exist = (evolving_mask[t - 1] == 1) & (evolving_mask[t] == 1)
            change_mask = (labels[t - 1] != labels[t]) & both_exist
            unchange_mask = (labels[t - 1] == labels[t]) & both_exist
        else:
            change_mask = np.zeros_like(m_t)
            unchange_mask = np.zeros_like(m_t)

        ch_true = labels[t][change_mask.astype(bool)]
        ch_pred = pred[change_mask.astype(bool)]
        groups["changed"].per_snapshot_count.append(len(ch_true))
        groups["changed"].per_snapshot_acc.append(compute_acc(ch_true, ch_pred))
        groups["changed"].per_snapshot_f1.append(compute_f1(ch_true, ch_pred, average=f1_average))
        groups["changed"].pooled_y_true.append(ch_true)
        groups["changed"].pooled_y_pred.append(ch_pred)

        un_true = labels[t][unchange_mask.astype(bool)]
        un_pred = pred[unchange_mask.astype(bool)]
        groups["unchanged"].per_snapshot_count.append(len(un_true))
        groups["unchanged"].per_snapshot_acc.append(compute_acc(un_true, un_pred))
        groups["unchanged"].per_snapshot_f1.append(compute_f1(un_true, un_pred, average=f1_average))
        groups["unchanged"].pooled_y_true.append(un_true)
        groups["unchanged"].pooled_y_pred.append(un_pred)

    return groups


def summarize_groups(groups: Dict[str, GroupStats], f1_average: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, g in groups.items():
        out[key] = {
            "support": float(g.support()),
            "snapshot_mean_acc": g.mean_acc(),
            "snapshot_mean_f1": g.mean_f1(),
            "pooled_acc": g.pooled_acc(),
            "pooled_f1": g.pooled_f1(f1_average),
        }
    return out
