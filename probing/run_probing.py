#!/usr/bin/env python3
"""Multi-dataset linear probing for frozen LVCG embeddings.

For each ``(dataset, label_ratio, model)`` triple the backbone runs once to
cache features; a small linear head is trained per seed on cached embeddings.

Usage::

    python run_probing.py --config ../configs/eval/probing.yaml
    python run_probing.py --config ../configs/eval/probing.yaml \
        --models lvcg --dataset ptbxl_super_class --ratio 0.1 --seed 42
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


from datasets import DatasetBundle, create_provider  # noqa: E402
from encoders import create_encoder  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset config normalisation
# ---------------------------------------------------------------------------

def _normalize_dataset_entries(cfg: dict) -> Dict[str, dict]:
    """Convert ``cfg['datasets']`` into a uniform ``{name: provider_cfg}`` map.

    A flat list of benchmark names plus a global ``data:`` block is expanded
    into :class:`BenchmarkProvider` configs.  Full dict entries (e.g. AI-READI)
    pass through unchanged.
    """
    raw = cfg["datasets"]
    data_defaults = cfg.get("data", {}) or {}
    chapman_classes = cfg.get("chapman_classes")
    repo_root = PACKAGE_ROOT

    out: Dict[str, dict] = {}

    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str):
                out[entry] = {
                    "type": "benchmark",
                    "dataset_name": entry,
                    "raw_root": data_defaults.get("raw_root"),
                    "splits_root": data_defaults.get(
                        "splits_root", str(repo_root / "data_splits")
                    ),
                    "norm_method": data_defaults.get("norm_method", "zscore"),
                    "chapman_classes": (
                        chapman_classes if entry == "chapman" else None
                    ),
                }
            elif isinstance(entry, dict) and len(entry) == 1:
                (name, body), = entry.items()
                out[name] = body
            else:
                raise ValueError(f"Unsupported dataset entry: {entry!r}")
    elif isinstance(raw, dict):
        out = dict(raw)
    else:
        raise ValueError(f"cfg['datasets'] must be list or dict, got {type(raw)}")

    return out


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading (delegates to a registered DatasetProvider)
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_name: str,
    label_ratio: float,
    batch_size: int,
    num_workers: int,
    provider_cfg: Optional[dict] = None,
):
    """Build train/val/test loaders via a :class:`DatasetProvider`.

    Training subsets use ``random_state=42`` inside :class:`BenchmarkECGDataset`
    so a given ``label_ratio`` is deterministic across probe seeds.
    """
    if provider_cfg is None:
        raise ValueError(
            "provider_cfg is required; set datasets in probing.yaml"
        )
    provider = create_provider(dataset_name, provider_cfg)
    bundle = provider.build(
        label_ratio=label_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return bundle.as_tuple()


# ---------------------------------------------------------------------------
# Embedding pre-extraction (backbone runs ONCE, not every epoch)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    all_embs, all_labels = [], []
    for batch in tqdm(loader, desc="    extract", leave=False):
        ecg = batch["ecg"].to(device)
        emb = encoder.ext_ecg_emb(ecg)
        all_embs.append(emb.cpu())
        all_labels.append(batch["label"])
    return torch.cat(all_embs), torch.cat(all_labels)


def make_emb_loader(
    embs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    ds = TensorDataset(embs, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


# ---------------------------------------------------------------------------
# Linear head
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Training on cached embeddings
# ---------------------------------------------------------------------------

def train_linear_head(
    head: LinearHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 5,
) -> float:
    head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = 0.0
    patience_counter = 0
    best_state = None

    pbar = tqdm(range(max_epochs), desc="  probe", leave=False)
    for _ in pbar:
        head.train()
        for embs, labels in train_loader:
            embs, labels = embs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            loss = criterion(head(embs), labels)
            loss.backward()
            optimizer.step()

        val_auroc = _compute_auroc_cached(head, val_loader, device)
        pbar.set_postfix(val_auc=f"{val_auroc:.4f}", best=f"{best_val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return best_val_auroc


def _compute_auroc_cached(
    head: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    head.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for embs, labels in loader:
            logits = head(embs.to(device))
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels)
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    try:
        return float(roc_auc_score(all_labels, all_probs, average="macro"))
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# Multi-metric evaluation
# ---------------------------------------------------------------------------

def _collect_probs_labels(
    head: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    head.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for embs, labels in loader:
            logits = head(embs.to(device))
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def _find_optimal_thresholds(
    probs: np.ndarray, labels: np.ndarray, grid_size: int = 51
) -> np.ndarray:
    n_classes = probs.shape[1]
    thresholds = np.linspace(0.05, 0.95, grid_size)
    best_thresh = np.full(n_classes, 0.5)
    for c in range(n_classes):
        best_f1 = -1.0
        for t in thresholds:
            p = (probs[:, c] > t).astype(int)
            f = f1_score(labels[:, c], p, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thresh[c] = t
    return best_thresh


def compute_all_metrics(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
) -> Dict[str, float]:
    """Compute AUROC, F1, Accuracy, Precision, Recall on cached embeddings.

    If ``val_loader`` is provided, per-class optimal thresholds are tuned on the
    validation set and applied to the test set.  Otherwise we fall back to 0.5.
    """
    probs, labels = _collect_probs_labels(head, loader, device)

    if val_loader is not None:
        val_probs, val_labels = _collect_probs_labels(head, val_loader, device)
        thresholds = _find_optimal_thresholds(val_probs, val_labels)
    else:
        thresholds = np.full(probs.shape[1], 0.5)

    preds = (probs > thresholds[np.newaxis, :]).astype(int)

    try:
        auroc = float(roc_auc_score(labels, probs, average="macro"))
    except ValueError:
        auroc = 0.5

    return {
        "auroc": auroc,
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(
            precision_score(labels, preds, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(labels, preds, average="macro", zero_division=0)
        ),
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "model", "dataset", "label_ratio", "seed", "source", "num_classes",
    "auroc", "f1", "accuracy", "precision", "recall",
    "val_auroc", "train_samples", "timestamp",
]


def init_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def append_csv(path: Path, row: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)


def already_done(
    path: Path, model: str, dataset: str, ratio: float, seed: int
) -> bool:
    if not path.exists():
        return False
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["model"] == model
                and row["dataset"] == dataset
                and float(row["label_ratio"]) == ratio
                and int(row["seed"]) == seed
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_seeds_from_cache(
    model_name: str,
    model_source: str,
    emb_dim: int,
    num_classes: int,
    train_embs: torch.Tensor,
    train_labels: torch.Tensor,
    val_embs: torch.Tensor,
    val_labels: torch.Tensor,
    test_embs: torch.Tensor,
    test_labels: torch.Tensor,
    dataset_name: str,
    label_ratio: float,
    seeds: List[int],
    probe_cfg: dict,
    device: torch.device,
    csv_path: Path,
) -> None:
    batch_size = probe_cfg.get("batch_size", 128)
    val_loader = make_emb_loader(val_embs, val_labels, batch_size)
    test_loader = make_emb_loader(test_embs, test_labels, batch_size)

    for seed in seeds:
        if already_done(csv_path, model_name, dataset_name, label_ratio, seed):
            print(f"      seed={seed} — already done, skipping")
            continue

        t0 = time.time()
        set_all_seeds(seed)
        train_loader = make_emb_loader(
            train_embs, train_labels, batch_size, shuffle=True
        )

        head = LinearHead(emb_dim, num_classes)
        val_auroc = train_linear_head(
            head, train_loader, val_loader, device,
            lr=probe_cfg.get("lr", 1e-3),
            max_epochs=probe_cfg.get("max_epochs", 100),
            patience=probe_cfg.get("patience", 5),
        )
        metrics = compute_all_metrics(head, test_loader, device, val_loader=val_loader)
        elapsed = time.time() - t0

        row = {
            "model": model_name,
            "dataset": dataset_name,
            "label_ratio": label_ratio,
            "seed": seed,
            "source": model_source,
            "num_classes": num_classes,
            "val_auroc": f"{val_auroc:.6f}",
            "train_samples": len(train_embs),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **{k: f"{v:.6f}" for k, v in metrics.items()},
        }
        append_csv(csv_path, row)

        print(
            f"      seed={seed}  AUROC={metrics['auroc']:.4f}  F1={metrics['f1']:.4f}  "
            f"Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  ({elapsed:.1f}s)"
        )
        del head


def main() -> None:
    parser = argparse.ArgumentParser(description="LVCG linear probing evaluation")
    parser.add_argument("--config", type=str, default="configs/probing.yaml")
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["lvcg"],
        help="Models to evaluate (subset of keys in cfg['models'])",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--results", type=str, default="results/probing_results.csv",
        help="CSV path (relative to package root or absolute)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PACKAGE_ROOT / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    csv_path = Path(args.results)
    if not csv_path.is_absolute():
        csv_path = PACKAGE_ROOT / csv_path
    init_csv(csv_path)

    all_models = cfg["models"]
    models = {name: all_models[name] for name in args.models if name in all_models}
    if not models:
        raise ValueError(
            f"None of {args.models} found in config. Available: {list(all_models)}"
        )
    dataset_map = _normalize_dataset_entries(cfg)
    if args.dataset is not None:
        if args.dataset not in dataset_map:
            raise ValueError(
                f"Dataset '{args.dataset}' not found in config; available: "
                f"{list(dataset_map)}"
            )
        dataset_map = {args.dataset: dataset_map[args.dataset]}
    ratios = cfg["label_ratios"] if args.ratio is None else [args.ratio]
    seeds = cfg["seeds"] if args.seed is None else [args.seed]

    total_probes = len(dataset_map) * len(ratios) * len(seeds) * len(models)
    total_extractions = len(dataset_map) * len(ratios) * len(models)
    print(
        f"Total probe runs: {total_probes}  "
        f"(embedding extractions: {total_extractions})"
    )

    for dataset_name, provider_cfg in dataset_map.items():
        for label_ratio in ratios:
            train_loader, val_loader, test_loader, num_classes = load_dataset(
                dataset_name=dataset_name,
                label_ratio=label_ratio,
                batch_size=cfg["probe"]["batch_size"],
                num_workers=cfg["probe"]["num_workers"],
                provider_cfg=provider_cfg,
            )

            print(
                f"\n{'='*60}\n"
                f"Dataset: {dataset_name} | Ratio: {label_ratio}\n"
                f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} "
                f"| Test: {len(test_loader.dataset)} | Classes: {num_classes}\n"
                f"{'='*60}"
            )

            for model_name, model_config in models.items():
                all_done = all(
                    already_done(csv_path, model_name, dataset_name, label_ratio, s)
                    for s in seeds
                )
                if all_done:
                    print(f"  {model_name} — all {len(seeds)} seeds done, skipping")
                    continue

                print(f"  {model_name}: extracting embeddings ...")
                t0 = time.time()
                try:
                    encoder = create_encoder(model_config, device=str(device))

                    train_embs, train_labels = extract_embeddings(
                        encoder, train_loader, device
                    )
                    val_embs, val_labels = extract_embeddings(
                        encoder, val_loader, device
                    )
                    test_embs, test_labels = extract_embeddings(
                        encoder, test_loader, device
                    )

                    emb_dim = encoder.out_features
                    del encoder
                    torch.cuda.empty_cache()
                    print(f"    done ({time.time() - t0:.1f}s, dim={emb_dim})")

                    run_seeds_from_cache(
                        model_name=model_name,
                        model_source=model_config.get("source", "rerun"),
                        emb_dim=emb_dim,
                        num_classes=num_classes,
                        train_embs=train_embs,
                        train_labels=train_labels,
                        val_embs=val_embs,
                        val_labels=val_labels,
                        test_embs=test_embs,
                        test_labels=test_labels,
                        dataset_name=dataset_name,
                        label_ratio=label_ratio,
                        seeds=seeds,
                        probe_cfg=cfg["probe"],
                        device=device,
                        csv_path=csv_path,
                    )
                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    for seed in seeds:
                        if not already_done(
                            csv_path, model_name, dataset_name, label_ratio, seed
                        ):
                            append_csv(csv_path, {
                                "model": model_name,
                                "dataset": dataset_name,
                                "label_ratio": label_ratio,
                                "seed": seed,
                                "source": model_config.get("source", "rerun"),
                                "num_classes": -1,
                                "auroc": -1, "f1": -1, "accuracy": -1,
                                "precision": -1, "recall": -1,
                                "val_auroc": -1,
                                "train_samples": -1,
                                "timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            })

    print(f"\nAll done! Results at: {csv_path}")


if __name__ == "__main__":
    main()
