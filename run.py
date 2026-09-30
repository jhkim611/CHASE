#!/usr/bin/env python3
"""CHASE training entry point. See README for usage."""

import argparse
import copy
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import (
    build_temporal_dataset,
    evaluate_test_groups,
    load_dataset,
    resolve_split,
    set_seed,
    summarize_groups,
    to_tensors,
)
from models import build_model, model_defaults


class ChangeAwareWrapper(nn.Module):
    """Wraps a single-step encoder with z_self, change-detection, and the change-aware classifier."""

    def __init__(
        self,
        base_model: nn.Module,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_classes: int = 2,
        n_structural: int = 2,
    ):
        super().__init__()
        self.base_model = base_model
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

        self.self_proj = nn.Linear(in_channels, hidden_channels)

        change_input_dim = hidden_channels + n_structural + 1  # h + signals + hom_proxy
        self.change_head = nn.Sequential(
            nn.Linear(change_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        cls_dim = hidden_channels * 2 + num_classes + 1  # h + z_self + prev_pred + p_change
        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

        self.prev_pred: Optional[torch.Tensor] = None
        self._p_change: Optional[torch.Tensor] = None
        self._z_self: Optional[torch.Tensor] = None

    def reset_state(self):
        self.base_model.reset_state()
        self.prev_pred = None

    def detach_state(self):
        self.base_model.detach_state()

    def encode(self, x, edge_index, edge_attr):
        h = self.base_model.encode(x, edge_index, edge_attr)
        self._z_self = F.relu(self.self_proj(x))
        return h

    @torch.no_grad()
    def _compute_homophily_prev_proxy(self, edge_index_prev, N, device):
        if self.prev_pred is None or edge_index_prev is None or edge_index_prev.shape[1] == 0:
            return torch.zeros(N, device=device)
        node_class = self.prev_pred.argmax(dim=1)
        src, dst = edge_index_prev[0], edge_index_prev[1]
        neighbor_agreement = self.prev_pred[src].gather(1, node_class[dst].unsqueeze(1)).squeeze(1)
        hom_proxy = torch.zeros(N, device=device)
        degree = torch.zeros(N, device=device)
        hom_proxy.scatter_add_(0, dst, neighbor_agreement)
        degree.scatter_add_(0, dst, torch.ones_like(neighbor_agreement))
        degree = degree.clamp(min=1)
        return hom_proxy / degree

    def classify(self, h, structural=None, edge_index_prev=None):
        N = h.shape[0]
        device = h.device

        parts = [h, self._z_self]

        hom_proxy = self._compute_homophily_prev_proxy(edge_index_prev, N, device)
        if structural is not None:
            ch_input = torch.cat([h, structural, hom_proxy.unsqueeze(1)], dim=1)
        else:
            ch_input = torch.cat([h, torch.zeros(N, 2, device=device),
                                  hom_proxy.unsqueeze(1)], dim=1)
        p_change = torch.sigmoid(self.change_head(ch_input))
        self._p_change = p_change.squeeze(-1)
        parts.append(p_change)

        if self.prev_pred is not None:
            parts.append(self.prev_pred)
        else:
            parts.append(torch.ones(N, self.num_classes, device=device) / self.num_classes)

        return self.classifier(torch.cat(parts, dim=1))

    @torch.no_grad()
    def store_predictions(self, logits):
        if logits.shape[-1] == 1 or logits.dim() == 1:
            p = torch.sigmoid(logits.squeeze(-1))
            self.prev_pred = torch.stack([1 - p, p], dim=1)
        else:
            self.prev_pred = F.softmax(logits, dim=1)

    def predict_change_logits(self, h, structural, edge_index_prev=None):
        N = h.shape[0]
        device = h.device
        hom_proxy = self._compute_homophily_prev_proxy(edge_index_prev, N, device)
        ch_input = torch.cat([h, structural, hom_proxy.unsqueeze(1)], dim=1)
        return self.change_head(ch_input).squeeze(-1)


def compute_structural_signals(feats, edge_indices, mask, t, device):
    N = feats.shape[1]
    signals = torch.zeros(N, 2, device=device)
    if t == 0:
        return signals
    ei_t = edge_indices[t]
    ei_prev = edge_indices[t - 1]
    adj_t = {}
    for e in range(ei_t.shape[1]):
        u, v = int(ei_t[0, e]), int(ei_t[1, e])
        adj_t.setdefault(u, set()).add(v)
        adj_t.setdefault(v, set()).add(u)
    adj_prev = {}
    for e in range(ei_prev.shape[1]):
        u, v = int(ei_prev[0, e]), int(ei_prev[1, e])
        adj_prev.setdefault(u, set()).add(v)
        adj_prev.setdefault(v, set()).add(u)
    for n in range(N):
        if not mask[t, n] or not mask[t - 1, n]:
            continue
        self_drift = float(np.linalg.norm(feats[t, n] - feats[t - 1, n]))
        nb_t = [j for j in adj_t.get(n, set()) if mask[t, j]]
        nb_prev = [j for j in adj_prev.get(n, set()) if mask[t - 1, j]]
        if nb_t and nb_prev:
            mean_curr = np.mean([feats[t, j] for j in nb_t], axis=0)
            mean_prev = np.mean([feats[t - 1, j] for j in nb_prev], axis=0)
            nb_shift = float(np.linalg.norm(mean_curr - mean_prev))
        else:
            nb_shift = 0.0
        signals[n] = torch.tensor([self_drift, nb_shift])
    return signals


def homophily_contrastive_loss(
    emb, labels, prev_labels, changed_mask, unchanged_mask,
    margin=0.5, homophily_weights=None,
):
    """Pull shifters toward the new-class centroid, scaled by (1 - homophily) so harder shifters weigh more."""
    if not changed_mask.any() or not unchanged_mask.any():
        return torch.tensor(0.0, device=emb.device)
    emb_norm = F.normalize(emb, dim=1)
    unique_labels = labels[unchanged_mask].unique()
    centroids = {}
    for lbl in unique_labels:
        lbl_mask = unchanged_mask & (labels == lbl.item())
        cent = emb_norm[lbl_mask].mean(dim=0)
        centroids[lbl.item()] = F.normalize(cent, dim=0)
    changed_idx = changed_mask.nonzero(as_tuple=True)[0]
    new_labs = labels[changed_idx]
    old_labs = prev_labels[changed_idx]
    valid = [i for i, (nl, ol) in enumerate(zip(new_labs, old_labs))
             if nl.item() in centroids and ol.item() in centroids]
    if not valid:
        return torch.tensor(0.0, device=emb.device)
    valid_t = torch.tensor(valid, device=emb.device, dtype=torch.long)
    c_emb = emb_norm[changed_idx[valid_t]]
    new_cents = torch.stack([centroids[new_labs[i].item()] for i in valid])
    old_cents = torch.stack([centroids[old_labs[i].item()] for i in valid])
    sim_new = (c_emb * new_cents).sum(dim=1)
    sim_old = (c_emb * old_cents).sum(dim=1)
    per_node_loss = F.relu(margin - (sim_new - sim_old))
    if homophily_weights is not None:
        hw = homophily_weights[changed_idx[valid_t]]
        hw = hw / hw.mean().clamp(min=1e-6)
        per_node_loss = per_node_loss * hw
    return per_node_loss.mean()


def change_detection_loss(change_logits, changed_mask, active_mask):
    active = active_mask.bool()
    if not active.any():
        return torch.tensor(0.0, device=change_logits.device)
    targets = changed_mask[active].float()
    logits = change_logits[active]
    n_pos = targets.sum().clamp(min=1)
    n_neg = (1 - targets).sum().clamp(min=1)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=n_neg / n_pos)


def get_masks(labels, mask, t):
    active = mask[t].astype(bool)
    if t == 0:
        return active, np.zeros_like(active), active.copy()
    prev_active = mask[t - 1].astype(bool)
    both = active & prev_active
    changed = both & (labels[t] != labels[t - 1])
    unchanged = both & (labels[t] == labels[t - 1])
    return active, changed, unchanged


def logits_to_pred(logits):
    if logits.dim() == 1 or logits.shape[-1] == 1:
        return (torch.sigmoid(logits.squeeze(-1)) > 0.5).long().cpu().numpy()
    return logits.argmax(dim=-1).cpu().numpy()


def compute_cls_loss(logits, y, active_mask, num_classes):
    active = active_mask.bool()
    if not active.any():
        return torch.tensor(0.0, device=logits.device)
    if num_classes <= 2:
        return F.binary_cross_entropy_with_logits(
            logits[active].squeeze(-1), y[active], reduction='mean')
    return F.cross_entropy(logits[active], y[active].long(), reduction='mean')


def compute_cls_loss_weighted(logits, y, active_mask, num_classes, sample_weight):
    active = active_mask.bool()
    if not active.any():
        return torch.tensor(0.0, device=logits.device)
    w = sample_weight[active]
    if num_classes <= 2:
        per_node = F.binary_cross_entropy_with_logits(
            logits[active].squeeze(-1), y[active], reduction='none')
    else:
        per_node = F.cross_entropy(logits[active], y[active].long(), reduction='none')
    return (per_node * w).mean()


def _get_chase_cls_loss(logits, y, active_t, changed_t, num_classes, device, N):
    if not changed_t.any():
        return compute_cls_loss(logits, y, active_t, num_classes)
    n_ch = changed_t.sum().clamp(min=1).float()
    n_unch = (active_t.sum() - n_ch).clamp(min=1).float()
    w_ch = torch.sqrt((n_ch + n_unch) / (2 * n_ch))
    w_unch = torch.sqrt((n_ch + n_unch) / (2 * n_unch))
    sw = torch.ones(N, device=device) * w_unch
    sw[changed_t.bool()] = w_ch
    return compute_cls_loss_weighted(logits, y, active_t, num_classes, sw)


def run_experiment(args):
    device = torch.device(args.device)
    set_seed(args.seed)

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    ds = load_dataset(args.dataset, Path(args.data_dir))
    snapshots = build_temporal_dataset(ds)
    K = len(snapshots)
    N = ds.feats.shape[1]
    F_dim = ds.feats.shape[2]
    num_classes = int(ds.labels.max()) + 1
    f1_avg = "binary" if num_classes <= 2 else "macro"
    out_ch = 1 if num_classes <= 2 else num_classes

    train_T, val_T, test_T = resolve_split(ds.name, K, args.train_T, args.val_T)
    test_start = train_T + val_T

    defaults = model_defaults()
    lr = args.lr if args.lr else defaults.lr
    hidden = args.hidden if args.hidden else defaults.hidden_channels

    is_baseline = not args.use_chase
    print(f"Dataset: {ds.name}  K={K} N={N} F={F_dim} classes={num_classes}")
    print(f"Backbone: GCN-GRU  +CHASE: {args.use_chase}")
    print(f"Split: train={train_T} val={val_T} test={test_T}")

    base_model = build_model(F_dim, hidden, num_classes=num_classes).to(device)
    emb_dim = base_model.gru.input_size if hasattr(base_model.gru, 'input_size') else hidden

    if is_baseline:
        # Replace the linear output head with the same MLP head CHASE uses
        # (fair-comparison classifier capacity).
        base_model.linear = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_ch),
        ).to(device)
        model = base_model
    else:
        model = ChangeAwareWrapper(
            base_model=base_model,
            in_channels=F_dim,
            hidden_channels=emb_dim,
            out_channels=out_ch,
            num_classes=num_classes,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    precomputed_structural = {}
    precomputed_hw = {}
    edge_index_tensors = {t: torch.tensor(ds.edge_indices[t], dtype=torch.long, device=device)
                          for t in range(K)}

    if args.use_chase:
        print("Precomputing structural signals...", end=" ", flush=True)
        for t in range(K):
            precomputed_structural[t] = compute_structural_signals(
                ds.feats, ds.edge_indices, ds.mask, t, device)
        print(f"done ({K} snapshots)")

        print("Precomputing homophily weights...", end=" ", flush=True)
        for t in range(1, train_T):
            pw = torch.ones(N, device=device)
            ei_t = ds.edge_indices[t]
            adj = {}
            for e in range(ei_t.shape[1]):
                u, v = int(ei_t[0, e]), int(ei_t[1, e])
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            for n in range(N):
                if not ds.mask[t, n] or not ds.mask[t - 1, n]:
                    continue
                if ds.labels[t, n] == ds.labels[t - 1, n]:
                    continue
                neighbors = adj.get(n, set())
                if not neighbors:
                    continue
                curr_label = int(ds.labels[t, n])
                nb_active = [j for j in neighbors if ds.mask[t, j]]
                if not nb_active:
                    continue
                match = sum(1 for j in nb_active if int(ds.labels[t, j]) == curr_label)
                hom = match / len(nb_active)
                pw[n] = 1.0 - hom
            precomputed_hw[t] = pw
        print(f"done ({len(precomputed_hw)} snapshots)")

    best_val, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.reset_state()
        epoch_loss = 0.0

        for t in range(train_T):
            model.detach_state()
            x, edge_index, edge_attr, y = to_tensors(snapshots[t], device)
            active_np, changed_np, unchanged_np = get_masks(ds.labels, ds.mask, t)
            active_t = torch.tensor(active_np, device=device)
            changed_t = torch.tensor(changed_np, device=device)
            unchanged_t = torch.tensor(unchanged_np, device=device)
            structural = precomputed_structural.get(t)

            optimizer.zero_grad()

            if is_baseline:
                logits = base_model(x, edge_index, edge_attr)
                loss = compute_cls_loss(logits, y, active_t, num_classes)
            else:
                emb = model.encode(x, edge_index, edge_attr)
                ei_prev = edge_index_tensors.get(t - 1) if t > 0 else None
                logits = model.classify(emb, structural, edge_index_prev=ei_prev)
                model.store_predictions(logits)

                loss = _get_chase_cls_loss(logits, y, active_t, changed_t, num_classes,
                                           device, x.shape[0])

                if t > 0 and changed_t.any() and unchanged_t.any():
                    labels_t = torch.tensor(ds.labels[t], dtype=torch.long, device=device)
                    prev_labels_t = torch.tensor(ds.labels[t - 1], dtype=torch.long, device=device)
                    loss_hom = homophily_contrastive_loss(
                        emb, labels_t, prev_labels_t, changed_t, unchanged_t,
                        margin=args.contrastive_margin,
                        homophily_weights=precomputed_hw.get(t),
                    )
                    loss = loss + args.lambda_hom * loss_hom

                if t > 0:
                    cl = model.predict_change_logits(emb, structural, edge_index_prev=ei_prev)
                    loss = loss + args.lambda_change * change_detection_loss(cl, changed_t, active_t)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        model.reset_state()
        val_correct, val_total = 0, 0
        val_ch_correct, val_ch_total = 0, 0
        val_unch_correct, val_unch_total = 0, 0

        with torch.no_grad():
            for t in range(train_T + val_T):
                model.detach_state()
                x, edge_index, edge_attr, y = to_tensors(snapshots[t], device)
                if is_baseline:
                    logits = base_model(x, edge_index, edge_attr)
                else:
                    emb = model.encode(x, edge_index, edge_attr)
                    ei_prev = edge_index_tensors.get(t - 1) if t > 0 else None
                    structural = precomputed_structural.get(t)
                    logits = model.classify(emb, structural, edge_index_prev=ei_prev)
                    model.store_predictions(logits)
                if t >= train_T:
                    pred = logits_to_pred(logits)
                    m = ds.mask[t].astype(bool)
                    val_correct += (pred[m] == ds.labels[t][m]).sum()
                    val_total += m.sum()
                    if t > 0:
                        both = m & ds.mask[t-1].astype(bool)
                        ch = both & (ds.labels[t] != ds.labels[t-1])
                        unch = both & (ds.labels[t] == ds.labels[t-1])
                        val_ch_correct += (pred[ch] == ds.labels[t][ch]).sum()
                        val_ch_total += ch.sum()
                        val_unch_correct += (pred[unch] == ds.labels[t][unch]).sum()
                        val_unch_total += unch.sum()

        val_all = val_correct / val_total if val_total > 0 else 0
        val_changed = val_ch_correct / val_ch_total if val_ch_total > 0 else 0

        if val_all > best_val:
            best_val = val_all
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % args.eval_every == 0:
            print(f"  epoch {epoch:4d}  loss={epoch_loss / max(train_T, 1):.4f}  "
                  f"val_all={val_all:.4f}  val_ch={val_changed:.4f}  bad={bad_epochs}")

        if bad_epochs >= args.patience:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model.reset_state()
    test_preds = []

    with torch.no_grad():
        for t in range(test_start + test_T):
            model.detach_state()
            x, edge_index, edge_attr, _ = to_tensors(snapshots[t], device)
            if is_baseline:
                emb = base_model.encode(x, edge_index, edge_attr)
                cls_head = base_model.linear
                logits = cls_head(emb)
            else:
                emb = model.encode(x, edge_index, edge_attr)
                ei_prev = edge_index_tensors.get(t - 1) if t > 0 else None
                structural = precomputed_structural.get(t)
                logits = model.classify(emb, structural, edge_index_prev=ei_prev)
                model.store_predictions(logits)
            if t >= test_start:
                test_preds.append(logits_to_pred(logits))

    groups = evaluate_test_groups(ds.labels, ds.mask, test_preds, test_start, f1_avg)
    summary = summarize_groups(groups, f1_avg)

    print(f"\n{'Group':<12} {'Acc':>8} {'F1':>8} {'Support':>8}")
    print("-" * 40)
    for g in ["all", "changed", "unchanged"]:
        s = summary[g]
        print(f"{g:<12} {s['pooled_acc']:>8.4f} {s['pooled_f1']:>8.4f} {s['support']:>8.0f}")

    model_name = "gcngru_chase" if not is_baseline else "gcngru_baseline"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.results_filename
    file_exists = csv_path.exists()
    fieldnames = [
        "dataset", "model", "seed", "group",
        "pooled_acc", "pooled_f1", "support",
        "snapshot_mean_acc", "snapshot_mean_f1",
        "use_chase",
        "lambda_hom", "lambda_change", "contrastive_margin", "f1_average",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for g in ["all", "changed", "unchanged"]:
            s = summary[g]
            writer.writerow({
                "dataset": ds.name, "model": model_name,
                "seed": args.seed, "group": g,
                "pooled_acc": round(s['pooled_acc'], 4),
                "pooled_f1": round(s['pooled_f1'], 4),
                "support": s['support'],
                "snapshot_mean_acc": round(s.get('snapshot_mean_acc', 0), 4),
                "snapshot_mean_f1": round(s.get('snapshot_mean_f1', 0), 4),
                "use_chase": args.use_chase,
                "lambda_hom": args.lambda_hom,
                "lambda_change": args.lambda_change,
                "contrastive_margin": args.contrastive_margin,
                "f1_average": f1_avg,
            })
    print(f"Results appended to {csv_path}")

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset name (e.g. DBLP) or path to a .pkl file")
    p.add_argument("--data-dir", type=str, default="main_files")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--train-T", type=int, default=None)
    p.add_argument("--val-T", type=int, default=None)
    p.add_argument("--use-chase", action="store_true",
                   help="Enable CHASE: contrastive loss, change-detection head, "
                        "and change-aware classifier (all components on).")
    p.add_argument("--lambda-hom", type=float, default=0.1)
    p.add_argument("--lambda-change", type=float, default=0.1)
    p.add_argument("--contrastive-margin", type=float, default=0.5)
    p.add_argument("--results-filename", type=str, default="results.csv")
    p.add_argument("--num-threads", type=int, default=0,
                   help="Limit CPU threads (0 = no limit)")
    args = p.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
