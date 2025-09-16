from __future__ import annotations

"""
med3pipe.vision.v3d.vision3d

Modular 3D volumetric classification for DenseNet121 (3D) and Swin Transformer (3D)
using NIfTI volumes prepared under SAM-Med3D folders:

SAM-Med3D-main/SAM-Med3D-main/
  data/train/<category>/<ct_name>/imagesTr/*.nii.gz
  data/validation/<category>/<ct_name>/imagesVal/*.nii.gz

Labels come from the dataset CSV (e.g., gist/sheet.csv) via med3pipe.sam.core.load_labels_from_sheet.

APIs:
- train_eval_densenet121_3d(paths, sheet_csv, ...)
- train_eval_swin_transformer_3d(paths, sheet_csv, ...)

Both functions allow selecting device ("cuda" or "cpu"; auto if None) for training and inference.
They save a run directory with:
- config.json
- best_model.pth
- predictions_val.csv (per-case)
- metrics.json
- classification_report.txt

Return a dict with objects and important paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import datetime as _dt
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import torchio as tio

from ...data.prepare import Sam3DPaths
from ...sam.core import load_labels_from_sheet, make_pre_transform

# Import MONAI nets independently so one failing import (e.g., SwinTransformer missing in older MONAI)
# doesn't disable the other (e.g., DenseNet121).
DenseNet121 = None  # type: ignore
SwinTransformer = None  # type: ignore
_DENSENET121_IMPORT_ERROR: Optional[Exception] = None
_SWIN_IMPORT_ERROR: Optional[Exception] = None
ViT = None  # type: ignore
_VIT_IMPORT_ERROR: Optional[Exception] = None
try:  # DenseNet121 is widely available across MONAI versions
    from monai.networks.nets import DenseNet121 as _MONAI_DenseNet121  # type: ignore
    DenseNet121 = _MONAI_DenseNet121  # type: ignore
except Exception as e:  # pragma: no cover
    _DENSENET121_IMPORT_ERROR = e
try:  # SwinTransformer location varies across MONAI releases
    from monai.networks.nets import SwinTransformer as _MONAI_SwinTransformer  # type: ignore
    SwinTransformer = _MONAI_SwinTransformer  # type: ignore
except Exception as e1:  # pragma: no cover
    try:
        from monai.networks.nets.swin_transformer import SwinTransformer as _MONAI_SwinTransformer  # type: ignore
        SwinTransformer = _MONAI_SwinTransformer  # type: ignore
    except Exception as e2:  # pragma: no cover
        _SWIN_IMPORT_ERROR = e2
try:  # ViT is the recommended MONAI transformer for classification
    from monai.networks.nets import ViT as _MONAI_ViT  # type: ignore
    ViT = _MONAI_ViT  # type: ignore
except Exception as e1:  # pragma: no cover
    try:
        from monai.networks.nets.vit import ViT as _MONAI_ViT  # type: ignore
        ViT = _MONAI_ViT  # type: ignore
    except Exception as e2:  # pragma: no cover
        _VIT_IMPORT_ERROR = e2


# ------------------------------
# Utils
# ------------------------------

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _device(auto: Optional[str] = None) -> torch.device:
    if auto is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(auto)


def _normalize_case_id_from_filename(fname: str) -> str:
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return Path(fname).stem


def _resolve_sheet_csv(sheet_csv: Optional[Path], paths: Sam3DPaths, dataset_root: Optional[Path]) -> Path:
    """Resolve sheet.csv path with sensible fallbacks.

    Preference order:
    1) explicit sheet_csv if exists
    2) dataset_root/sheet.csv if dataset_root provided
    3) CWD/gist/sheet.csv
    4) <project_root>/gist/sheet.csv where project_root is sam3d_root/../../
    """
    if sheet_csv is not None:
        p = Path(sheet_csv)
        if p.exists():
            return p
    candidates: List[Path] = []
    if dataset_root is not None:
        candidates.append(Path(dataset_root) / "sheet.csv")
    candidates.extend([
        Path.cwd() / "gist" / "sheet.csv",
        paths.sam3d_root.parent.parent / "gist" / "sheet.csv",
    ])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"sheet.csv not found. Tried: {[str(c) for c in candidates]} and explicit {sheet_csv}. "
        "Pass an absolute path to sheet_csv or specify dataset_root."
    )


# ------------------------------
# Dataset
# ------------------------------

def _to_logits(output: Any) -> torch.Tensor:
    """Normalize various model outputs to a logits tensor.

    Supports:
    - Tensor
    - Tuple/List where first element is Tensor (e.g., (logits, hidden))
    - Dict with 'logits' key
    """
    if isinstance(output, (list, tuple)) and len(output) > 0:
        first = output[0]
        if torch.is_tensor(first):
            return first
    if isinstance(output, dict) and 'logits' in output:
        v = output['logits']
        if torch.is_tensor(v):
            return v
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Model forward returned unsupported type for logits: {type(output)}")

class VolumeDataset(Dataset[Tuple[torch.Tensor, int, str, str]]):
    """3D NIfTI dataset for volume classification.

    Returns (volume, label, path, case_id), where volume has shape (C=1, D, H, W).
    """

    def __init__(
        self,
        img_dir: Path,
        lab_map: Dict[str, int],
        img_size: int = 96,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.img_dir = Path(img_dir)
        self.lab_map = lab_map
        self.img_size = img_size
        self.augment = augment
        self.paths: List[Path] = []
        # collect NIfTI files that have labels
        for ext in ("*.nii.gz", "*.nii"):
            for p in sorted(self.img_dir.glob(ext)):
                cid = _normalize_case_id_from_filename(p.name)
                if cid in self.lab_map:
                    self.paths.append(p)
        if len(self.paths) == 0:
            # Build a helpful error message
            all_files: List[Path] = []
            for ext in ("*.nii.gz", "*.nii"):
                all_files.extend(sorted(self.img_dir.glob(ext)))
            sample_cids = [_normalize_case_id_from_filename(p.name) for p in all_files[:10]]
            lab_keys = list(self.lab_map.keys())[:10]
            raise ValueError(
                "No labeled volumes found under "
                f"{self.img_dir}. This usually means case IDs from files do not match label map keys.\n"
                f"- Example case IDs on disk (first {len(sample_cids)}): {sample_cids}\n"
                f"- Example keys in label map (first {len(lab_keys)}): {lab_keys}\n"
                "Check dataset_name, subject_col, label_col, case_suffix, and sheet_csv path."
            )
        # transforms
        self.base_transform: tio.Compose = make_pre_transform(img_size=img_size)
        self.aug_transform: Optional[tio.Compose] = None
        if augment:
            self.aug_transform = tio.Compose(
                [
                    # light geometric intensity aggs suitable for classification
                    tio.RandomFlip(axes=(0, 1, 2)),
                    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=0),
                ]
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        p = self.paths[idx]
        cid = _normalize_case_id_from_filename(p.name)
        label = int(self.lab_map[cid])
        # load volume
        sitk_img = sitk.ReadImage(str(p))
        arr, _ = tio.data.io.sitk_to_nib(sitk_img)  # (1, D, H, W)
        subj = tio.Subject(image=tio.ScalarImage(tensor=arr))
        subj = self.base_transform(subj)
        if self.augment and self.aug_transform is not None:
            subj = self.aug_transform(subj)
        vol = subj.image.data.clone().detach()  # (1, D, H, W)
        return vol, label, str(p), cid


# ------------------------------
# Models (3D)
# ------------------------------

def build_densenet121_3d(num_classes: int = 2, in_channels: int = 1) -> nn.Module:
    if DenseNet121 is None:
        raise ImportError("MONAI is required for 3D DenseNet121. Install with `pip install monai`. ")
    # MONAI DenseNet121 supports spatial_dims argument
    model = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=num_classes)
    return model


def build_vit_3d(
    num_classes: int = 2,
    in_channels: int = 1,
    img_size: Tuple[int, int, int] = (96, 96, 96),
    patch_size: Tuple[int, int, int] = (16, 16, 16),
    hidden_size: int = 384,
    mlp_dim: int = 1536,
    num_layers: int = 8,
    num_heads: int = 6,
    pos_embed: str = "conv",
    dropout_rate: float = 0.0,
) -> nn.Module:
    if ViT is None:
        raise ImportError(
            "MONAI ViT is required for 3D ViT classification. Install/upgrade MONAI. "
            f"Import error: {_VIT_IMPORT_ERROR}"
        )
    try:
        model = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=True,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=3,
        )
    except TypeError:
        # Fallback for MONAI builds that don't expose `pos_embed` in ViT signature
        model = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=True,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=3,
        )
    return model


def build_swin_transformer_3d(
    num_classes: int = 2,
    in_channels: int = 1,
    img_size: Tuple[int, int, int] = (96, 96, 96),
    patch_size: Tuple[int, int, int] = (4, 4, 4),
    window_size: Tuple[int, int, int] = (7, 7, 7),
    embed_dim: int = 96,
    depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
    num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
) -> nn.Module:
    if SwinTransformer is None:
        raise ImportError("MONAI is required for 3D Swin Transformer. Install with `pip install monai`. ")
    model = SwinTransformer(
        in_chans=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        window_size=window_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        spatial_dims=3,
    )
    return model


# ------------------------------
# Train/Eval loop
# ------------------------------

@dataclass
class Train3DConfig:
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    img_size: int = 96
    batch_size: int = 2
    num_workers: int = 0
    augment: bool = True
    device: Optional[str] = None  # "cuda"/"cpu" or None for auto
    seed: int = 42


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _evaluate_3d(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    proba_list: List[np.ndarray] = []
    case_ids: List[str] = []
    paths: List[str] = []
    softmax = nn.Softmax(dim=1)
    for vol, label, path, cid in dl:
        # vol: (B, C=1, D, H, W)
        vol = vol.to(device, non_blocking=True)
        label_np = label.numpy().tolist()
        logits_raw = model(vol)
        logits = _to_logits(logits_raw)
        prob = softmax(logits).cpu().numpy()
        pred = prob.argmax(axis=1).tolist()
        y_true.extend(label_np)
        y_pred.extend(pred)
        proba_list.extend(list(prob))
        paths.extend(list(path))
        case_ids.extend(list(cid))
    # Metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report_txt = classification_report(y_true, y_pred, target_names=["0", "1"])  # binary default
    cm = confusion_matrix(y_true, y_pred).astype(int).tolist()
    roc_auc = None
    try:
        proba_arr = np.array(proba_list)
        if proba_arr.ndim == 2 and proba_arr.shape[1] == 2:
            roc_auc = float(roc_auc_score(y_true, proba_arr[:, 1]))
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
        "y_proba": proba_list,
        "case_ids": case_ids,
        "paths": paths,
    }


def _save_run_outputs(
    out_dir: Path,
    config: Dict[str, Any],
    eval_res: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    # Predictions per case
    import pandas as pd
    df = {
        "case_id": eval_res["case_ids"],
        "true_label": eval_res["y_true"],
        "pred_label": eval_res["y_pred"],
    }
    proba = np.array(eval_res["y_proba"]) if eval_res["y_proba"] else np.empty((0, 0))
    if proba.size > 0:
        for i in range(proba.shape[1]):
            df[f"proba_{i}"] = proba[:, i].tolist()
    pd.DataFrame(df).to_csv(out_dir / "predictions_val.csv", index=False)
    # Metrics
    (out_dir / "metrics.json").write_text(json.dumps({
        "accuracy": eval_res["acc"],
        "macro_f1": eval_res["macro_f1"],
        "roc_auc": eval_res["roc_auc"],
        "confusion_matrix": eval_res["cm"],
    }, indent=2))
    (out_dir / "classification_report.txt").write_text(eval_res["report"])


def _train_one_epoch(model: nn.Module, dl: DataLoader, device: torch.device, optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for vol, label, _, _ in dl:
        vol = vol.to(device, non_blocking=True)
        label = label.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits_raw = model(vol)
        logits = _to_logits(logits_raw)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * vol.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
    return running_loss / max(1, total), correct / max(1, total)


def _fit_model(
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
    history: List[Dict[str, Any]] = []
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, dl_train, device, optimizer, criterion)
        val_res = _evaluate_3d(model, dl_val, device)
        val_acc = val_res["acc"]
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_acc": best_acc, "history": history}


# ------------------------------
# Public APIs
# ------------------------------

def _build_dataloaders_from_paths(
    paths: Sam3DPaths,
    lab_map: Dict[str, int],
    img_size: int,
    batch_size: int,
    num_workers: int,
    augment_train: bool,
) -> Tuple[DataLoader, DataLoader]:
    ds_tr = VolumeDataset(paths.images_tr, lab_map=lab_map, img_size=img_size, augment=augment_train)
    ds_va = VolumeDataset(paths.images_val, lab_map=lab_map, img_size=img_size, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_tr, dl_va


def train_eval_densenet121_3d(
    paths: Sam3DPaths,
    sheet_csv: Optional[Path] = None,
    # Labels
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    dataset_root: Optional[Path] = None,
    # Model/data/training
    num_classes: int = 2,
    cfg: Optional[Train3DConfig] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    img_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    augment: Optional[bool] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = cfg or Train3DConfig()
    if epochs is not None: cfg.epochs = epochs
    if lr is not None: cfg.lr = lr
    if weight_decay is not None: cfg.weight_decay = weight_decay
    if img_size is not None: cfg.img_size = img_size
    if batch_size is not None: cfg.batch_size = batch_size
    if num_workers is not None: cfg.num_workers = num_workers
    if augment is not None: cfg.augment = augment
    if device is not None: cfg.device = device

    _set_seed(cfg.seed)
    dev = _device(cfg.device)

    # labels (resolve sheet_csv with fallbacks)
    sheet_csv_resolved = _resolve_sheet_csv(sheet_csv, paths, dataset_root)
    _, lab_map = load_labels_from_sheet(
        sheet_csv=sheet_csv_resolved,
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
    )
    dl_tr, dl_va = _build_dataloaders_from_paths(paths, lab_map, cfg.img_size, cfg.batch_size, cfg.num_workers, cfg.augment)

    model = build_densenet121_3d(num_classes=num_classes, in_channels=1)
    model, fit_info = _fit_model(model, dl_tr, dl_va, dev, cfg.epochs, cfg.lr, cfg.weight_decay)

    # Evaluate best on val
    eval_res = _evaluate_3d(model, dl_va, dev)

    out_dir = Path.cwd() / "vision3d_runs" / f"densenet121_3d_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pth")
    _save_run_outputs(
        out_dir,
        {
            "model": "densenet121_3d",
            "num_classes": num_classes,
            "train_config": vars(cfg),
        },
        eval_res,
    )
    return {
        "model": model,
        "out_dir": out_dir,
        "eval": eval_res,
        "fit": fit_info,
        "model_path": out_dir / "best_model.pth",
    }


def train_eval_vit_3d(
    paths: Sam3DPaths,
    sheet_csv: Optional[Path] = None,
    # Labels
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    dataset_root: Optional[Path] = None,
    # Model/data/training
    num_classes: int = 2,
    img_size_3d: Tuple[int, int, int] = (96, 96, 96),
    patch_size: Tuple[int, int, int] = (16, 16, 16),
    hidden_size: int = 384,
    mlp_dim: int = 1536,
    num_layers: int = 8,
    num_heads: int = 6,
    pos_embed: str = "conv",
    dropout_rate: float = 0.0,
    cfg: Optional[Train3DConfig] = None,
    # inline overrides
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    img_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    augment: Optional[bool] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = cfg or Train3DConfig()
    if epochs is not None: cfg.epochs = epochs
    if lr is not None: cfg.lr = lr
    if weight_decay is not None: cfg.weight_decay = weight_decay
    if img_size is not None: cfg.img_size = img_size
    if batch_size is not None: cfg.batch_size = batch_size
    if num_workers is not None: cfg.num_workers = num_workers
    if augment is not None: cfg.augment = augment
    if device is not None: cfg.device = device

    _set_seed(cfg.seed)
    dev = _device(cfg.device)

    sheet_csv_resolved = _resolve_sheet_csv(sheet_csv, paths, dataset_root)
    _, lab_map = load_labels_from_sheet(
        sheet_csv=sheet_csv_resolved,
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
    )
    dl_tr, dl_va = _build_dataloaders_from_paths(paths, lab_map, cfg.img_size, cfg.batch_size, cfg.num_workers, cfg.augment)

    model = build_vit_3d(
        num_classes=num_classes,
        in_channels=1,
        img_size=img_size_3d,
        patch_size=patch_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        pos_embed=pos_embed,
        dropout_rate=dropout_rate,
    )
    model, fit_info = _fit_model(model, dl_tr, dl_va, dev, cfg.epochs, cfg.lr, cfg.weight_decay)

    eval_res = _evaluate_3d(model, dl_va, dev)

    out_dir = Path.cwd() / "vision3d_runs" / f"vit3d_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pth")
    _save_run_outputs(
        out_dir,
        {
            "model": "vit3d",
            "num_classes": num_classes,
            "train_config": vars(cfg),
            "vit": {
                "img_size_3d": list(img_size_3d),
                "patch_size": list(patch_size),
                "hidden_size": hidden_size,
                "mlp_dim": mlp_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "pos_embed": pos_embed,
                "dropout_rate": dropout_rate,
            },
        },
        eval_res,
    )
    return {
        "model": model,
        "out_dir": out_dir,
        "eval": eval_res,
        "fit": fit_info,
        "model_path": out_dir / "best_model.pth",
    }


def train_eval_swin_transformer_3d(
    paths: Sam3DPaths,
    sheet_csv: Optional[Path] = None,
    # Labels
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    dataset_root: Optional[Path] = None,
    # Model/data/training
    num_classes: int = 2,
    img_size_3d: Tuple[int, int, int] = (96, 96, 96),
    patch_size: Tuple[int, int, int] = (4, 4, 4),
    window_size: Tuple[int, int, int] = (7, 7, 7),
    embed_dim: int = 96,
    depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
    num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
    cfg: Optional[Train3DConfig] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    img_size: Optional[int] = None,  # for preprocessing cube side length
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    augment: Optional[bool] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = cfg or Train3DConfig()
    if epochs is not None: cfg.epochs = epochs
    if lr is not None: cfg.lr = lr
    if weight_decay is not None: cfg.weight_decay = weight_decay
    if img_size is not None: cfg.img_size = img_size
    if batch_size is not None: cfg.batch_size = batch_size
    if num_workers is not None: cfg.num_workers = num_workers
    if augment is not None: cfg.augment = augment
    if device is not None: cfg.device = device

    _set_seed(cfg.seed)
    dev = _device(cfg.device)

    sheet_csv_resolved = _resolve_sheet_csv(sheet_csv, paths, dataset_root)
    _, lab_map = load_labels_from_sheet(
        sheet_csv=sheet_csv_resolved,
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
    )
    dl_tr, dl_va = _build_dataloaders_from_paths(paths, lab_map, cfg.img_size, cfg.batch_size, cfg.num_workers, cfg.augment)

    model = build_swin_transformer_3d(
        num_classes=num_classes,
        in_channels=1,
        img_size=img_size_3d,
        patch_size=patch_size,
        window_size=window_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
    )
    model, fit_info = _fit_model(model, dl_tr, dl_va, dev, cfg.epochs, cfg.lr, cfg.weight_decay)

    # Evaluate best on val
    eval_res = _evaluate_3d(model, dl_va, dev)

    out_dir = Path.cwd() / "vision3d_runs" / f"swin3d_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pth")
    _save_run_outputs(
        out_dir,
        {
            "model": "swin3d",
            "num_classes": num_classes,
            "train_config": vars(cfg),
            "swin": {
                "img_size_3d": list(img_size_3d),
                "patch_size": list(patch_size),
                "window_size": list(window_size),
                "embed_dim": embed_dim,
                "depths": list(depths),
                "num_heads": list(num_heads),
            },
        },
        eval_res,
    )
    return {
        "model": model,
        "out_dir": out_dir,
        "eval": eval_res,
        "fit": fit_info,
        "model_path": out_dir / "best_model.pth",
    }
