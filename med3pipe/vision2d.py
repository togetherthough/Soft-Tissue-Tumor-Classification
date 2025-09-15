from __future__ import annotations

"""
med3pipe.vision2d

Modular 2D image classification training for DenseNet121 and Swin Transformer
using ImageFolder datasets laid out as:

<dataset_root>/
  train/
    0/  ... .png
    1/  ... .png
  val/
    0/  ... .png
    1/  ... .png

Intended to work with datasets exported by med3pipe.export2d.export_swin_dataset
(which saves 3-channel PNG slices named like '<CASE>_z###.png').

APIs:
- train_eval_densenet121(...): Train/eval DenseNet121 on ImageFolder data.
- train_eval_swin_transformer(...): Train/eval Swin Transformer (timm) on ImageFolder data.

Both functions save a run directory with:
- config.json
- best_model.pth
- class_to_idx.json
- slice_predictions.csv (one row per image)
- case_predictions.csv (aggregated by CASE prefix)
- metrics_slice.json, metrics_case.json
- classification_report_slice.txt, classification_report_case.txt

Return a dict with objects and important paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import datetime as _dt
import json
import math
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import timm  # for Swin Transformer
except Exception:  # pragma: no cover - optional at import time
    timm = None  # type: ignore


# ------------------------------
# Utils
# ------------------------------

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_out_dir(model_name: str) -> Path:
    out = Path.cwd() / "vision2d_runs" / f"{model_name}_{_timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _device(auto: Optional[str] = None) -> torch.device:
    if auto is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(auto)


def _build_transforms(img_size: int, augment: str = "light") -> Tuple[transforms.Compose, transforms.Compose]:
    # Use ImageNet normalization to match pretrained weights
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if augment == "none":
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        # light augmentations by default
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def _create_dataloaders(
    data_root: Path,
    img_size: int,
    batch_size: int,
    num_workers: int = 0,
    augment: str = "light",
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    train_tf, val_tf = _build_transforms(img_size, augment=augment)
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected 'train' and 'val' under {data_root}")
    ds_train = ImageFolderWithPaths(train_dir, transform=train_tf)
    ds_val = ImageFolderWithPaths(val_dir, transform=val_tf)
    class_to_idx = ds_train.class_to_idx
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, class_to_idx


# ------------------------------
# Models
# ------------------------------

def build_densenet121(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    from torchvision import models
    # Handle torchvision API differences
    weights = None
    try:
        from torchvision.models import DenseNet121_Weights  # type: ignore
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
    except Exception:
        model = models.densenet121(pretrained=pretrained)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, num_classes)
    return model


def build_swin_transformer(
    model_name: str = "swin_tiny_patch4_window7_224",
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    if timm is None:
        raise ImportError("timm is required for Swin Transformer models. Install with `pip install timm`. ")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


# ------------------------------
# Train/Eval loop
# ------------------------------

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    augment: str = "light"  # or "none"
    device: Optional[str] = None  # "cuda"/"cpu" or None for auto
    seed: int = 42
    case_agg: str = "mean"  # aggregation of slice probabilities per case


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _case_id_from_path(p: str) -> str:
    # Expect filenames like '<CASE>_z###.png' -> return '<CASE>'
    name = Path(p).name
    m = re.match(r"(.+)_z\d+\.(?:png|jpg|jpeg)$", name, flags=re.IGNORECASE)
    return m.group(1) if m else Path(p).stem


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    class_to_idx: Dict[str, int],
) -> Dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[np.ndarray] = []
    paths: list[str] = []
    softmax = nn.Softmax(dim=1)
    for batch in dl:
        # batch is (images, targets, paths)
        if len(batch) == 3:
            images, targets, batch_paths = batch
        else:
            images, targets = batch
            batch_paths = [None] * len(targets)
        images = images.to(device, non_blocking=True)
        targets_np = targets.numpy().tolist()
        logits = model(images)
        prob = softmax(logits).cpu().numpy()
        pred = prob.argmax(axis=1).tolist()
        y_true.extend(targets_np)
        y_pred.extend(pred)
        y_proba.extend(list(prob))
        if batch_paths[0] is not None:
            paths.extend([str(p) for p in batch_paths])
    # Metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report_txt = classification_report(y_true, y_pred, target_names=[k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])])
    cm = confusion_matrix(y_true, y_pred).astype(int).tolist()
    # ROC-AUC if binary and prob available
    roc_auc = None
    try:
        if len(class_to_idx) == 2:
            proba_1 = np.array([p[1] for p in y_proba])
            roc_auc = float(roc_auc_score(y_true, proba_1))
    except Exception:
        roc_auc = None
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "report": report_txt,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "paths": paths,
    }


def _aggregate_case_level(slice_eval: Dict[str, Any], agg: str = "mean") -> Dict[str, Any]:
    # Aggregate slice probabilities per case using mean or max
    paths = slice_eval["paths"]
    y_true = slice_eval["y_true"]
    y_proba = slice_eval["y_proba"]
    assert len(paths) == len(y_true) == len(y_proba)
    case_to_probs: Dict[str, list[Tuple[int, np.ndarray]]] = {}
    for p, yt, pr in zip(paths, y_true, y_proba):
        cid = _case_id_from_path(p)
        case_to_probs.setdefault(cid, []).append((yt, np.asarray(pr)))
    case_ids: list[str] = []
    y_true_c: list[int] = []
    y_pred_c: list[int] = []
    y_proba_c: list[np.ndarray] = []
    for cid, items in case_to_probs.items():
        case_ids.append(cid)
        # assume all true labels identical per case; fallback to majority
        labels = [it[0] for it in items]
        if len(set(labels)) == 1:
            yt = labels[0]
        else:
            # majority vote
            vals, cnts = np.unique(labels, return_counts=True)
            yt = int(vals[np.argmax(cnts)])
        probs = np.stack([it[1] for it in items], axis=0)
        if agg == "max":
            p = probs.max(axis=0)
        else:
            p = probs.mean(axis=0)
        y_true_c.append(yt)
        y_proba_c.append(p)
        y_pred_c.append(int(p.argmax()))
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
    acc = float(accuracy_score(y_true_c, y_pred_c))
    macro_f1 = float(f1_score(y_true_c, y_pred_c, average="macro"))
    report_txt = classification_report(y_true_c, y_pred_c)
    cm = confusion_matrix(y_true_c, y_pred_c).astype(int).tolist()
    roc_auc = None
    try:
        if y_proba_c and y_proba_c[0].shape[0] == 2:
            roc_auc = float(roc_auc_score(y_true_c, np.array(y_proba_c)[:, 1]))
    except Exception:
        roc_auc = None
    return {
        "case_ids": case_ids,
        "y_true": y_true_c,
        "y_pred": y_pred_c,
        "y_proba": [p.tolist() for p in y_proba_c],
        "acc": acc,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "report": report_txt,
        "cm": cm,
    }


def _save_run_outputs(
    out_dir: Path,
    model_name: str,
    config: Dict[str, Any],
    class_to_idx: Dict[str, int],
    slice_eval: Dict[str, Any],
    case_eval: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2))
    # Slice-level predictions
    import pandas as pd
    proba_arr = np.array(slice_eval["y_proba"]) if slice_eval["y_proba"] else np.empty((0, len(class_to_idx)))
    df_slice = {
        "path": slice_eval["paths"],
        "true_label": slice_eval["y_true"],
        "pred_label": slice_eval["y_pred"],
    }
    if proba_arr.size > 0 and proba_arr.shape[1] >= 2:
        for i in range(proba_arr.shape[1]):
            df_slice[f"proba_{i}"] = proba_arr[:, i].tolist()
    pd.DataFrame(df_slice).to_csv(out_dir / "slice_predictions.csv", index=False)
    (out_dir / "metrics_slice.json").write_text(json.dumps({
        "accuracy": slice_eval["acc"],
        "macro_f1": slice_eval["macro_f1"],
        "roc_auc": slice_eval["roc_auc"],
        "confusion_matrix": slice_eval["cm"],
    }, indent=2))
    (out_dir / "classification_report_slice.txt").write_text(slice_eval["report"])
    # Case-level predictions
    df_case = {
        "case_id": case_eval["case_ids"],
        "true_label": case_eval["y_true"],
        "pred_label": case_eval["y_pred"],
    }
    if case_eval.get("y_proba"):
        proba_case = np.array(case_eval["y_proba"])  # list -> array
        if proba_case.ndim == 2:
            for i in range(proba_case.shape[1]):
                df_case[f"proba_{i}"] = proba_case[:, i].tolist()
    pd.DataFrame(df_case).to_csv(out_dir / "case_predictions.csv", index=False)
    (out_dir / "metrics_case.json").write_text(json.dumps({
        "accuracy": case_eval["acc"],
        "macro_f1": case_eval["macro_f1"],
        "roc_auc": case_eval["roc_auc"],
        "confusion_matrix": case_eval["cm"],
    }, indent=2))
    (out_dir / "classification_report_case.txt").write_text(case_eval["report"])


def _train_one_epoch(model: nn.Module, dl: DataLoader, device: torch.device, optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in dl:
        if len(batch) == 3:
            images, targets, _ = batch
        else:
            images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def _fit_classifier(
    model: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = -math.inf
    best_state = None
    history = []
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, dl_train, device, optimizer, criterion)
        # quick eval
        eval_res = _evaluate(model, dl_val, device, class_to_idx=dl_val.dataset.class_to_idx)  # type: ignore[attr-defined]
        val_acc = eval_res["acc"]
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_acc": best_acc, "history": history}


# ------------------------------
# Public APIs
# ------------------------------

def train_eval_densenet121(
    data_root: Path,
    out_dir: Optional[Path] = None,
    num_classes: int = 2,
    cfg: Optional[TrainConfig] = None,
    # inline hyperparams for convenience
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    img_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    augment: Optional[str] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    case_agg: Optional[str] = None,
    pretrained: bool = True,
) -> Dict[str, Any]:
    """Train/Eval DenseNet121 on an ImageFolder dataset.

    You can pass a TrainConfig via `cfg` or individual overrides.
    """
    model_name = "densenet121"
    cfg = cfg or TrainConfig()
    if epochs is not None: cfg.epochs = epochs
    if lr is not None: cfg.lr = lr
    if weight_decay is not None: cfg.weight_decay = weight_decay
    if img_size is not None: cfg.img_size = img_size
    if batch_size is not None: cfg.batch_size = batch_size
    if num_workers is not None: cfg.num_workers = num_workers
    if augment is not None: cfg.augment = augment
    if device is not None: cfg.device = device
    if seed is not None: cfg.seed = seed
    if case_agg is not None: cfg.case_agg = case_agg

    _set_seed(cfg.seed)
    device_t = _device(cfg.device)
    dl_train, dl_val, class_to_idx = _create_dataloaders(Path(data_root), cfg.img_size, cfg.batch_size, cfg.num_workers, cfg.augment)
    model = build_densenet121(num_classes=num_classes, pretrained=pretrained)
    model, fit_info = _fit_classifier(model, dl_train, dl_val, device_t, cfg.epochs, cfg.lr, cfg.weight_decay)

    # Evaluate best on val
    slice_eval = _evaluate(model, dl_val, device_t, class_to_idx)
    case_eval = _aggregate_case_level(slice_eval, agg=cfg.case_agg)

    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir(model_name)
    # Save model and outputs
    torch.save(model.state_dict(), out_dir / "best_model.pth")
    _save_run_outputs(
        out_dir,
        model_name,
        {
            "model": model_name,
            "num_classes": num_classes,
            "train_config": vars(cfg),
            "fit": fit_info,
        },
        class_to_idx,
        slice_eval,
        case_eval,
    )

    return {
        "model": model,
        "out_dir": out_dir,
        "class_to_idx": class_to_idx,
        "slice_eval": slice_eval,
        "case_eval": case_eval,
        "fit": fit_info,
        "model_path": out_dir / "best_model.pth",
    }


# ------------------------------
# Dataset with path return
# ------------------------------

class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder that returns (image, target, path)."""
    def __getitem__(self, index):  # type: ignore[override]
        sample, target = super().__getitem__(index)
        path = self.samples[index][0]
        return sample, target, path


def train_eval_swin_transformer(
    data_root: Path,
    out_dir: Optional[Path] = None,
    num_classes: int = 2,
    model_name: str = "swin_tiny_patch4_window7_224",
    cfg: Optional[TrainConfig] = None,
    # inline hyperparams
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    img_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    augment: Optional[str] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    case_agg: Optional[str] = None,
    pretrained: bool = True,
) -> Dict[str, Any]:
    """Train/Eval a Swin Transformer (from timm) on an ImageFolder dataset.

    Set model_name to one of timm's Swin variants, e.g.:
    - 'swin_tiny_patch4_window7_224'
    - 'swin_small_patch4_window7_224'
    - 'swin_base_patch4_window7_224'
    """
    if timm is None:
        raise ImportError("timm is required. Install with `pip install timm`. ")

    base_name = model_name
    cfg = cfg or TrainConfig()
    if epochs is not None: cfg.epochs = epochs
    if lr is not None: cfg.lr = lr
    if weight_decay is not None: cfg.weight_decay = weight_decay
    if img_size is not None: cfg.img_size = img_size
    if batch_size is not None: cfg.batch_size = batch_size
    if num_workers is not None: cfg.num_workers = num_workers
    if augment is not None: cfg.augment = augment
    if device is not None: cfg.device = device
    if seed is not None: cfg.seed = seed
    if case_agg is not None: cfg.case_agg = case_agg

    _set_seed(cfg.seed)
    device_t = _device(cfg.device)
    dl_train, dl_val, class_to_idx = _create_dataloaders(Path(data_root), cfg.img_size, cfg.batch_size, cfg.num_workers, cfg.augment)
    model = build_swin_transformer(model_name=base_name, num_classes=num_classes, pretrained=pretrained)
    model, fit_info = _fit_classifier(model, dl_train, dl_val, device_t, cfg.epochs, cfg.lr, cfg.weight_decay)

    # Evaluate best on val
    slice_eval = _evaluate(model, dl_val, device_t, class_to_idx)
    case_eval = _aggregate_case_level(slice_eval, agg=cfg.case_agg)

    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir(base_name)
    # Save model and outputs
    torch.save(model.state_dict(), out_dir / "best_model.pth")
    _save_run_outputs(
        out_dir,
        base_name,
        {
            "model": base_name,
            "num_classes": num_classes,
            "train_config": vars(cfg),
            "fit": fit_info,
        },
        class_to_idx,
        slice_eval,
        case_eval,
    )

    return {
        "model": model,
        "out_dir": out_dir,
        "class_to_idx": class_to_idx,
        "slice_eval": slice_eval,
        "case_eval": case_eval,
        "fit": fit_info,
        "model_path": out_dir / "best_model.pth",
    }
