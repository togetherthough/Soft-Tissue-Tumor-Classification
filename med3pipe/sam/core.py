from __future__ import annotations

"""
med3pipe.sam.core

Methods for steps 4–6:
4) Build SAM-Med3D model and extract image-encoder embeddings for TRAIN and VAL.
5) ROI-pool embeddings with lesion masks to get per-case vectors.
6) Load labels from dataset sheets and align to case IDs.

These functions reuse the directory layout produced by med3pipe.data.prepare (steps 1–3):
- data/train/<category>/<ct_name>/{imagesTr,labelsTr}
- data/validation/<category>/<ct_name>/{imagesVal,labelsVal}
- features/<category>/{ct_name}_train and features/<category>/<ct_name>
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchio as tio
import SimpleITK as sitk

from ..data.prepare import Sam3DPaths, find_default_sam3d_root


# ------------------------------
# Model build and feature extraction (Step 4)
# ------------------------------

@dataclass
class Sam3DModelSpec:
    model_type: str = "vit_b_ori"
    checkpoint: Optional[Path] = None
    device: Optional[torch.device] = None
    img_size: int = 128


def _ensure_repo_on_path(sam3d_root: Path) -> None:
    sam3d_root = Path(sam3d_root)
    if str(sam3d_root) not in sys.path:
        sys.path.insert(0, str(sam3d_root))


def build_sam3d_model(
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    device: Optional[torch.device] = None,
    eval_mode: bool = True,
):
    """Construct a SAM-Med3D model and optionally load a checkpoint.

    - sam3d_root: path to SAM-Med3D repo inner root (auto-detected if None)
    - model_type: architecture key for `sam_model_registry3D`
    - checkpoint: path to .pth checkpoint (optional)
    - device: torch.device (defaults to cuda if available else cpu)
    - eval_mode: whether to set model.eval()
    """
    sam3d_root = sam3d_root or find_default_sam3d_root()
    _ensure_repo_on_path(sam3d_root)

    from segment_anything.build_sam3D import sam_model_registry3D  # type: ignore

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = sam_model_registry3D[model_type](checkpoint=None).to(device)
    if checkpoint is not None and Path(checkpoint).exists():
        ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print("Loaded checkpoint (model_state_dict)")
        except Exception:
            model.load_state_dict(ckpt, strict=False)
            print("Loaded checkpoint (raw state_dict)")
    if eval_mode:
        model.eval()
    else:
        model.train()
    return model


def _znorm_masking_method(x):
    """Masking method for ZNormalization. Defined at module level for pickling."""
    return x > 0


class ResizeLargestTo(tio.Transform):
    """Resize volume so that the largest dimension becomes target_size.
    
    This preserves aspect ratio and minimizes data loss compared to CropOrPad.
    Useful for preparing data for SAM-Med3D where we want 128^3 format.
    """
    def __init__(self, target_size: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Get the first image to determine spatial shape
        first_image = subject.get_first_image()
        spatial_shape = first_image.spatial_shape  # (D, H, W)
        
        # Find largest dimension
        max_dim = max(spatial_shape)
        
        # Calculate scale factor to make largest dimension = target_size
        if max_dim > 0:
            scale = self.target_size / max_dim
        else:
            scale = 1.0
        
        # Calculate new shape (all dimensions scaled proportionally)
        new_shape = tuple(int(dim * scale) for dim in spatial_shape)
        
        # Apply resize to all images in subject
        resize_transform = tio.Resize(target_shape=new_shape)
        subject = resize_transform(subject)
        
        return subject


def make_pre_transform(img_size: int = 128) -> tio.Compose:
    """Create preprocessing transform pipeline for SAM-Med3D.
    
    Uses resize-then-pad approach to minimize data loss:
    1. Resize so largest dimension becomes img_size (preserves aspect ratio)
    2. Pad to img_size^3 (adds minimal padding since largest dim is already correct)
    3. Z-normalize
    """
    return tio.Compose(
        [
            tio.ToCanonical(),
            ResizeLargestTo(target_size=img_size),
            tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
            tio.ZNormalization(masking_method=_znorm_masking_method),
        ]
    )


def load_volume_tensor(img_path: Path, pre_transform: Optional[Callable] = None) -> torch.Tensor:
    """Load a NIfTI volume and return a tensor of shape (1, 1, D, H, W)."""
    if pre_transform is None:
        pre_transform = make_pre_transform()
    sitk_img = sitk.ReadImage(str(img_path))
    sitk_arr, _ = tio.data.io.sitk_to_nib(sitk_img)  # (1, D, H, W)
    subject = tio.Subject(image=tio.ScalarImage(tensor=sitk_arr))
    subject = pre_transform(subject)
    image = subject.image.data.clone().detach()  # (1, D, H, W)
    image = image.unsqueeze(0)  # (1,1,D,H,W)
    return image


def extract_embeddings(
    model: torch.nn.Module,
    img_dir: Path,
    out_dir: Path,
    device: Optional[torch.device] = None,
    pre_transform: Optional[Callable] = None,
    skip_existing: bool = True,
) -> int:
    """Extract image-encoder embeddings for all *.nii.gz in img_dir and save to out_dir.

    Each output is a .pt file with keys: {"path": <nii>, "embedding": <tensor on cpu>}.
    Returns the number of embeddings written.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(img_dir).glob("*.nii.gz"))
    print("To extract:", len(paths), "from", img_dir)
    cnt = 0
    with torch.no_grad():
        for ipath in paths:
            out_pt = out_dir / (ipath.stem + "_embedding.pt")
            if skip_existing and out_pt.exists():
                continue
            vol = load_volume_tensor(ipath, pre_transform=pre_transform).to(device)
            emb = model.image_encoder(vol)
            # Validate embedding shape
            if emb.dim() != 5:
                print(f"[WARN] Unexpected embedding dims for {ipath.name}: {emb.shape} (expected 5D tensor)")
            torch.save({"path": str(ipath), "embedding": emb.cpu()}, str(out_pt))
            cnt += 1
            if cnt % 25 == 0:
                print(f"Extracted {cnt} embeddings ...")
    print("Done extracting to:", out_dir)
    return cnt


@dataclass
class FeatureDirs:
    train_dir: Path
    val_dir: Path


def default_feature_dirs(sam3d_root: Path, category: str, ct_name: str) -> FeatureDirs:
    sam3d_root = Path(sam3d_root)
    feat_train = sam3d_root / "features" / category / f"{ct_name}_train"
    feat_val = sam3d_root / "features" / category / ct_name
    feat_train.mkdir(parents=True, exist_ok=True)
    feat_val.mkdir(parents=True, exist_ok=True)
    return FeatureDirs(train_dir=feat_train, val_dir=feat_val)


def extract_embeddings_train_val(
    paths: Sam3DPaths,
    model: torch.nn.Module,
    sam3d_root: Optional[Path] = None,
    img_size: int = 128,
    feature_dirs: Optional[FeatureDirs] = None,
    device: Optional[torch.device] = None,
    skip_existing: bool = True,
) -> FeatureDirs:
    """Extract embeddings for TRAIN imagesTr and VAL imagesVal into default feature dirs.
    
    Args:
        skip_existing: If True (default), skip extraction for files that already exist.
                      Set to False to force re-extraction (e.g., after loading new checkpoint).
    """
    if sam3d_root is None:
        sam3d_root = find_default_sam3d_root()
    if feature_dirs is None:
        feature_dirs = default_feature_dirs(sam3d_root, category=paths.category, ct_name=paths.ct_name)

    pre_transform = make_pre_transform(img_size=img_size)
    extract_embeddings(model, paths.images_tr, feature_dirs.train_dir, device=device, pre_transform=pre_transform, skip_existing=skip_existing)
    extract_embeddings(model, paths.images_val, feature_dirs.val_dir, device=device, pre_transform=pre_transform, skip_existing=skip_existing)
    return feature_dirs


# ------------------------------
# ROI pooling (Step 5)
# ------------------------------

def load_mask_tensor(mask_path: Path, pre_transform: Optional[Callable] = None) -> torch.Tensor:
    """Load and preprocess a mask into a tensor of shape (1, D, H, W) with values in {0,1}."""
    if pre_transform is None:
        pre_transform = make_pre_transform()
    sitk_mask = sitk.ReadImage(str(mask_path))
    mask_arr, _ = tio.data.io.sitk_to_nib(sitk_mask)  # (1, D, H, W)
    subj = tio.Subject(label=tio.LabelMap(tensor=mask_arr))
    subj = pre_transform(subj)
    mask = (subj.label.data.clone().detach() > 0).float()
    return mask


def roi_pool_embedding(embedding: torch.Tensor, mask: Optional[torch.Tensor]) -> np.ndarray:
    """ROI pool a single embedding (1,C,d,h,w) by the provided mask (1,D,H,W).

    If mask is None or empty after downsampling, fallback is global average pooling.
    Returns a numpy array of shape (C,).
    """
    B, C, d, h, w = embedding.shape
    assert B == 1
    if mask is None:
        return embedding.mean(dim=(2, 3, 4)).squeeze(0).cpu().numpy()
    mask_ds = F.interpolate(mask.unsqueeze(0), size=(d, h, w), mode="nearest").squeeze(0)
    weights = (mask_ds > 0.5).float()
    wsum = weights.sum()
    if wsum < 1.0:
        return embedding.mean(dim=(2, 3, 4)).squeeze(0).cpu().numpy()
    pooled = (embedding * weights).sum(dim=(2, 3, 4)) / wsum
    return pooled.squeeze(0).cpu().numpy()


def case_id_from_pt(pt: Path) -> str:
    return pt.stem.replace("_embedding", "")


def load_roi_features(
    feat_dir: Path,
    label_dir: Optional[Path],
    pre_transform: Optional[Callable] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Load ROI pooled feature vectors from embedding .pt files in `feat_dir`.

    If `label_dir` is provided, masks will be used for ROI pooling; otherwise global average pooling.
    Returns (X, ids) where X is shape (N, C) and ids are case_id strings.
    """
    X: List[np.ndarray] = []
    ids: List[str] = []
    for pt in sorted(Path(feat_dir).glob("*_embedding.pt")):
        d = torch.load(str(pt), map_location="cpu", weights_only=False)
        emb = d["embedding"]
        if emb.dim() == 2:  # (1, C) -> add spatial dims
            emb = emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cid = case_id_from_pt(pt)
        m = None
        if label_dir is not None:
            mp = Path(label_dir) / f"{cid}.nii.gz"
            if mp.exists():
                try:
                    m = load_mask_tensor(mp, pre_transform=pre_transform)
                except Exception as e:
                    print(f"[WARN] mask load failed {cid}: {e}")
        feat = roi_pool_embedding(emb, m)
        X.append(feat)
        ids.append(cid)
    if not X:
        return np.empty((0,)), []
    # Check for shape consistency before stacking
    shapes = [x.shape for x in X]
    if len(set(shapes)) > 1:
        print(f"[ERROR] Shape mismatch in features:")
        for i, (cid, shape) in enumerate(zip(ids, shapes)):
            print(f"  {cid}: {shape}")
        raise ValueError(f"all input arrays must have the same shape\nTraining shapes: {set(shapes)}")
    return np.stack(X), ids


# ------------------------------
# Diagnostics (Embedding shapes before PCA)
# ------------------------------

def _embedding_shape_from_pt(pt: Path) -> Tuple[int, ...]:
    d = torch.load(str(pt), map_location="cpu", weights_only=False)
    emb = d["embedding"]
    return tuple(emb.shape)


def summarize_embedding_shapes_dir(feat_dir: Path) -> tuple[set[Tuple[int, ...]], list[tuple[str, Tuple[int, ...]]]]:
    """Return (unique_shapes, differing_files) for all *_embedding.pt in feat_dir."""
    uniq: set[Tuple[int, ...]] = set()
    diffs: list[tuple[str, Tuple[int, ...]]] = []
    pts = sorted(Path(feat_dir).glob("*_embedding.pt"))
    for pt in pts:
        shp = _embedding_shape_from_pt(pt)
        if uniq and shp not in uniq:
            diffs.append((pt.name, shp))
        uniq.add(shp)
    return uniq, diffs


def summarize_train_val_embedding_shapes(feature_dirs: FeatureDirs) -> dict:
    """Summarize embedding shapes for train/val feature dirs."""
    train_shapes, train_diff = summarize_embedding_shapes_dir(feature_dirs.train_dir)
    val_shapes, val_diff = summarize_embedding_shapes_dir(feature_dirs.val_dir)

    def channel_dims(shapes: set[Tuple[int, ...]]) -> set[int]:
        ch: set[int] = set()
        for s in shapes:
            if len(s) >= 2:
                ch.add(int(s[1]))
            elif len(s) == 1:
                ch.add(int(s[0]))
        return ch

    return {
        "train": {
            "dir": str(feature_dirs.train_dir),
            "unique_shapes": sorted([list(s) for s in train_shapes]),
            "channel_dims": sorted(list(channel_dims(train_shapes))),
            "diff_files": [(n, list(s)) for n, s in train_diff],
        },
        "val": {
            "dir": str(feature_dirs.val_dir),
            "unique_shapes": sorted([list(s) for s in val_shapes]),
            "channel_dims": sorted(list(channel_dims(val_shapes))),
            "diff_files": [(n, list(s)) for n, s in val_diff],
        },
    }

# ------------------------------
# Labels (Step 6)
# ------------------------------

def load_labels_from_sheet(
    sheet_csv: Path,
    dataset_col: str = "Dataset",
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    binarize_neg1_to0: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Load a CSV sheet and build a label map {<Subject><case_suffix>: label}.

    - If dataset_name is provided, filter rows where `dataset_col` equals that string.
    - Coerce `label_col` to int, returning 0 for invalid/missing.
    - If `binarize_neg1_to0` is True (default), map -1 -> 0 and clamp labels to {0,1}.
      This is a pragmatic quick fix for mixed/ambiguous target encodings across datasets
      (e.g., {-1,0,1} where -1 might mean "unknown" or "benign"). Binarizing ensures
      consistent targets for pooling and enables per‑dataset ROC AUC to be well‑defined.
      Disable if you explicitly want multi‑class behavior.
    - Return the filtered dataframe and a dict mapping case_id to int label.
    """
    df = pd.read_csv(sheet_csv)
    if dataset_name is not None and dataset_col in df.columns:
        df = df[df[dataset_col].astype(str).str.strip() == dataset_name].copy()
    if df.empty:
        raise ValueError("No rows found after filtering by dataset (if applied)")

    df[subject_col] = df[subject_col].astype(str).str.strip()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    lab = df[label_col].astype(int)
    if binarize_neg1_to0:
        # Map -1 -> 0, and clamp to {0,1}
        lab = lab.replace({-1: 0})
        lab = lab.clip(lower=0, upper=1)
    df["label"] = lab.astype(int)

    lab_map = {f"{sid}{case_suffix}": int(lbl) for sid, lbl in zip(df[subject_col], df["label"])}
    return df, lab_map


def build_y(ids: Sequence[str], lab_map: dict) -> Tuple[np.ndarray, list]:
    """Map ids to labels. Strips a trailing '.nii' if present to support case IDs from filenames."""
    y: List[int] = []
    missing: List[str] = []
    for cid in ids:
        clean_cid = cid.replace(".nii", "") if cid.endswith(".nii") else cid
        if clean_cid in lab_map:
            y.append(lab_map[clean_cid])
        else:
            missing.append(cid)
    return np.array(y, dtype=int), missing
