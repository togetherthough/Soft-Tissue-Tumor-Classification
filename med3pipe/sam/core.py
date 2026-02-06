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
from typing import Callable, List, Optional, Sequence, Tuple, Literal
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchio as tio
import SimpleITK as sitk

from ..data.prepare import Sam3DPaths, find_default_sam3d_root

# Type alias for pooling strategies
PoolingStrategy = Literal['avg', 'multiscale', 'percentile']


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
    use_medim: bool = True,
):
    """Construct a SAM-Med3D model and optionally load a checkpoint.

    - sam3d_root: path to SAM-Med3D repo inner root (auto-detected if None)
    - model_type: architecture key for `sam_model_registry3D`
    - checkpoint: path to .pth checkpoint (optional)
    - device: torch.device (defaults to cuda if available else cpu)
    - eval_mode: whether to set model.eval()
    - use_medim: if True (default), use medim.create_model for loading; otherwise use legacy method
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use medim for loading if requested and checkpoint is provided
    if use_medim and checkpoint is not None and Path(checkpoint).exists():
        try:
            import medim
            print(f"Loading SAM-Med3D model via medim from: {checkpoint}")
            model = medim.create_model(
                "SAM-Med3D",
                pretrained=True,
                checkpoint_path=str(checkpoint)
            ).to(device)
            if eval_mode:
                model.eval()
            else:
                model.train()
            print("✅ SAM-Med3D model loaded successfully via medim!")
            return model
        except ImportError:
            print("⚠️  medim not available, falling back to legacy loading method")
        except Exception as e:
            print(f"⚠️  medim loading failed ({e})")
            # If CUDA is not available, retry with CPU patching
            if not torch.cuda.is_available() and checkpoint is not None:
                print("🔧 Retrying with CPU map_location patch...")
                import torch as torch_module
                original_load = torch_module.load
                
                def patched_load(*args, **kwargs):
                    if 'map_location' not in kwargs:
                        kwargs['map_location'] = 'cpu'
                    return original_load(*args, **kwargs)
                
                torch_module.load = patched_load
                try:
                    model = medim.create_model(
                        "SAM-Med3D",
                        pretrained=True,
                        checkpoint_path=str(checkpoint)
                    ).to(device)
                    if eval_mode:
                        model.eval()
                    else:
                        model.train()
                    print("✅ SAM-Med3D model loaded successfully with CPU patch!")
                    return model
                except Exception as e2:
                    print(f"❌ CPU patch also failed: {e2}")
                finally:
                    torch_module.load = original_load
            
            # If all medim attempts fail, raise error (no legacy fallback available)
            raise RuntimeError(
                f"Failed to load SAM-Med3D model via medim: {e}\n"
                "Legacy loading method requires SAM-Med3D source code to be installed.\n"
                "Please ensure:\n"
                "  1. CUDA is available (if using GPU)\n"
                "  2. medim library is installed: pip install medim\n"
                "  3. Checkpoint file exists and is valid"
            )


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
    1. ToCanonical: Ensure consistent orientation (RAS+).
    2. CT Windowing: Clamp intensities to soft-tissue window [-150, 250] HU.
       This focuses the model on relevant tissue contrast and ignores extreme values (air/bone).
    3. ResizeLargestTo: Resize so largest dimension becomes img_size (preserves aspect ratio).
    4. CropOrPad: Pad to img_size^3 (adds minimal padding since largest dim is already correct).
    5. ZNormalization: Standardize mean=0, std=1 (computed over entire volume).
    """
    return tio.Compose(
        [
            tio.ToCanonical(),
            # CT Soft Tissue Windowing: [-150, 250]
            # Typical soft tissue is Level 40, Width 400 => -160 to 240.
            # We use a slightly wider range to capture all soft tissue.
            tio.Clamp(out_min=-150, out_max=250),
            ResizeLargestTo(target_size=img_size),
            tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
            # ZNormalization without masking (use all pixels, including fat/air in the windowed range)
            tio.ZNormalization(masking_method=None),
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
            # Handle .nii.gz extension properly: remove both .gz and .nii
            base_name = ipath.name.replace('.nii.gz', '')
            out_pt = out_dir / f"{base_name}_embedding.pt"
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


def pool_avg(embedding: torch.Tensor) -> np.ndarray:
    """Global Average Pooling (GAP).
    
    Args:
        embedding: Tensor of shape (B, C, D, H, W) or (1, C, D, H, W)
        
    Returns:
        Pooled vector of shape (C,)
    """
    return embedding.mean(dim=(2, 3, 4)).squeeze(0).cpu().numpy()


def pool_multiscale(embedding: torch.Tensor) -> np.ndarray:
    """Multiscale spatial pyramid pooling at 1×1×1, 2×2×2, and 4×4×4 scales.
    
    Args:
        embedding: Tensor of shape (B, C, D, H, W) or (1, C, D, H, W)
        
    Returns:
        Concatenated pooled vector of shape (C × 73,)
        where 73 = 1³ + 2³ + 4³ = 1 + 8 + 64
    """
    B, C, d, h, w = embedding.shape
    assert B == 1, "Batch size must be 1"
    
    pooled_features = []
    
    for scale in [1, 2, 4]:
        # Average pool to (scale, scale, scale) grid
        pooled = F.adaptive_avg_pool3d(embedding, output_size=(scale, scale, scale))
        # Flatten spatial dimensions: (1, C, scale, scale, scale) -> (C, scale^3)
        pooled = pooled.view(C, -1)
        # Flatten to (C * scale^3,)
        pooled = pooled.flatten()
        pooled_features.append(pooled.cpu().numpy())
    
    return np.concatenate(pooled_features)


def pool_percentile(embedding: torch.Tensor, percentiles=(10, 25, 50, 75, 90)) -> np.ndarray:
    """Percentile pooling across spatial dimensions.
    
    Args:
        embedding: Tensor of shape (B, C, D, H, W) or (1, C, D, H, W)
        percentiles: Tuple of percentile values to compute (default: 10, 25, 50, 75, 90)
        
    Returns:
        Pooled vector of shape (C × len(percentiles),)
    """
    B, C, d, h, w = embedding.shape
    assert B == 1, "Batch size must be 1"
    
    # Flatten spatial dimensions: (1, C, D, H, W) -> (C, D*H*W)
    emb_flat = embedding.view(C, -1).cpu().numpy()
    
    pooled = []
    for p in percentiles:
        # Compute percentile along spatial dimension (axis=1)
        pct = np.percentile(emb_flat, p, axis=1)  # Shape: (C,)
        pooled.append(pct)
    
    # Stack and flatten: (len(percentiles), C) -> (C * len(percentiles),)
    return np.concatenate(pooled)


# Dictionary mapping strategy names to functions
POOLING_STRATEGIES = {
    'avg': pool_avg,
    'multiscale': pool_multiscale,
    'percentile': pool_percentile,
}


def average_pool_embedding(
    embedding: torch.Tensor, 
    mask: Optional[torch.Tensor] = None,
    pooling_strategy: str = 'percentile'
) -> np.ndarray:
    """Pool a single embedding using the specified strategy.

    Args:
        embedding: Tensor of shape (B, C, D, H, W) or (1, C, D, H, W)
        mask: Optional mask tensor (currently ignored, kept for backward compatibility)
        pooling_strategy: Pooling strategy to use ('avg', 'multiscale', or 'percentile')
        
    Returns:
        Pooled feature vector. Shape depends on pooling strategy:
        - 'avg': (C,)
        - 'multiscale': (C × 73,)
        - 'percentile': (C × 5,)
    """
    if pooling_strategy not in POOLING_STRATEGIES:
        raise ValueError(
            f"Invalid pooling strategy '{pooling_strategy}'. "
            f"Valid options are: {list(POOLING_STRATEGIES.keys())}"
        )
    
    pool_fn = POOLING_STRATEGIES[pooling_strategy]
    return pool_fn(embedding)


def case_id_from_pt(pt: Path) -> str:
    return pt.stem.replace("_embedding", "")


def load_pooled_features(
    feat_dir: Path,
    label_dir: Optional[Path],
    pre_transform: Optional[Callable] = None,
    pooling_strategy: str = 'percentile',
) -> Tuple[np.ndarray, List[str]]:
    """Load feature vectors from embedding .pt files in `feat_dir`.

    Args:
        feat_dir: Directory containing embedding .pt files
        label_dir: Optional label directory (currently ignored, kept for backward compatibility)
        pre_transform: Optional preprocessing transform (currently ignored)
        pooling_strategy: Pooling strategy to use ('avg', 'multiscale', or 'percentile')
        
    Returns:
        Tuple of (X, ids) where:
        - X is shape (N, D) where D depends on pooling strategy
        - ids are case_id strings
    """
    X: List[np.ndarray] = []
    ids: List[str] = []
    for pt in sorted(Path(feat_dir).glob("*_embedding.pt")):
        d = torch.load(str(pt), map_location="cpu", weights_only=False)
        emb = d["embedding"]
        if emb.dim() == 2:  # (1, C) -> add spatial dims
            emb = emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cid = case_id_from_pt(pt)
        
        # Mask loading removed as we switched to GAP (kept here for compatibility)
        m = None
        
        feat = average_pool_embedding(emb, m, pooling_strategy=pooling_strategy)
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
