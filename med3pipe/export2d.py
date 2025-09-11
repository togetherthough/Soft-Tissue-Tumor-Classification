from __future__ import annotations

"""
med3pipe.export2d

Export 3D NIfTI cases (train/val) to 2D slice classification datasets for Swin Transformer.
- Produces ImageNet-style folders: <export_root>/{train,val}/{0,1}/<CASE>_z###.png
- Optionally select only slices intersecting lesion masks.
- Intensity windowing by percentiles per-volume and resize to img_size, saved as 3-channel PNG.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image

from .prepare import Sam3DPaths
from .sam3d import load_labels_from_sheet


@dataclass
class ExportStats:
    train_counts: Dict[int, int]
    val_counts: Dict[int, int]
    skipped_no_label: List[str]
    skipped_no_mask_slices: List[str]
    export_root: Path


def _volume_to_uint8(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def _save_slice_img(slice_arr: np.ndarray, out_path: Path, img_size: int = 224) -> None:
    # slice_arr: (H, W), uint8
    img = Image.fromarray(slice_arr)
    if img.mode != "L":
        img = img.convert("L")
    img = img.resize((img_size, img_size), resample=Image.BICUBIC)
    # make 3-channel by stacking
    img = Image.merge("RGB", (img, img, img))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _select_slices(mask_arr: np.ndarray, max_slices: Optional[int], seed: int) -> List[int]:
    # mask_arr: (Z, H, W) with values {0, 1}
    pos = [i for i in range(mask_arr.shape[0]) if mask_arr[i].sum() > 0]
    if not pos:
        return []
    if max_slices is None or len(pos) <= max_slices:
        return pos
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pos), size=max_slices, replace=False)
    return sorted([pos[i] for i in idx])


def _normalize_case_id_from_filename(fname: str) -> str:
    """Strip common extensions to recover case_id (e.g., 'GIST-001_CT' from 'GIST-001_CT.nii.gz')."""
    if fname.endswith('.nii.gz'):
        return fname[:-7]
    if fname.endswith('.nii'):
        return fname[:-4]
    return Path(fname).stem


def export_swin_dataset(
    paths: Sam3DPaths,
    sheet_csv: Path,
    export_root: Path,
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    img_size: int = 224,
    slices_per_case: Optional[int] = 64,
    only_masked: bool = True,
    seed: int = 2025,
) -> ExportStats:
    """Export 2D slices for Swin Transformer training from prepared SAM-Med3D paths.

    Returns ExportStats with counts and skipped cases.
    """
    export_root = Path(export_root)
    train_out = export_root / "train"
    val_out = export_root / "val"
    train_counts: Dict[int, int] = {0: 0, 1: 0}
    val_counts: Dict[int, int] = {0: 0, 1: 0}
    skipped_no_label: List[str] = []
    skipped_no_mask_slices: List[str] = []

    # Build label map
    _, lab_map = load_labels_from_sheet(
        sheet_csv,
        dataset_col="Dataset",
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
    )

    def process_split(img_dir: Path, lbl_dir: Path, out_dir: Path, counter: Dict[int, int]):
        for nii in sorted(img_dir.glob("*.nii.gz")):
            cid = _normalize_case_id_from_filename(nii.name)  # e.g., GIST-003_CT
            if cid not in lab_map:
                skipped_no_label.append(cid)
                continue
            label = int(lab_map[cid])
            # Read image and mask
            img_itk = sitk.ReadImage(str(nii))
            img_arr = sitk.GetArrayFromImage(img_itk)  # (Z, H, W)
            mask_path = lbl_dir / f"{cid}.nii.gz"
            if not mask_path.exists():
                # no mask? still export middle slices if only_masked False
                mask_arr = np.zeros_like(img_arr, dtype=np.uint8)
            else:
                m_itk = sitk.ReadImage(str(mask_path))
                mask_arr = (sitk.GetArrayFromImage(m_itk) > 0).astype(np.uint8)
                # if geometry differs, we simply crop/pad to match shapes
                if mask_arr.shape != img_arr.shape:
                    minz = min(mask_arr.shape[0], img_arr.shape[0])
                    miny = min(mask_arr.shape[1], img_arr.shape[1])
                    minx = min(mask_arr.shape[2], img_arr.shape[2])
                    img_arr = img_arr[:minz, :miny, :minx]
                    mask_arr = mask_arr[:minz, :miny, :minx]

            # select slices
            if only_masked:
                idxs = _select_slices(mask_arr, slices_per_case, seed)
                if not idxs:
                    skipped_no_mask_slices.append(cid)
                    continue
            else:
                Z = img_arr.shape[0]
                if slices_per_case is None or slices_per_case >= Z:
                    idxs = list(range(Z))
                else:
                    # uniform sampling across Z
                    idxs = list(np.linspace(0, Z - 1, num=slices_per_case, dtype=int))

            # windowing
            img_u8 = _volume_to_uint8(img_arr)
            for z in idxs:
                slice_img = img_u8[z]
                out_p = out_dir / str(label) / f"{cid}_z{z:03d}.png"
                _save_slice_img(slice_img, out_p, img_size=img_size)
                counter[label] = counter.get(label, 0) + 1

    process_split(paths.images_tr, paths.labels_tr, train_out, train_counts)
    process_split(paths.images_val, paths.labels_val, val_out, val_counts)

    return ExportStats(
        train_counts=train_counts,
        val_counts=val_counts,
        skipped_no_label=skipped_no_label,
        skipped_no_mask_slices=skipped_no_mask_slices,
        export_root=export_root,
    )
