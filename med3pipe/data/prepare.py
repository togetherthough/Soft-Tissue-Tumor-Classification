from __future__ import annotations

"""
med3pipe.data.prepare

Implements steps 1–3 of the notebook `notebooks/gist_tabpfn_end_to_end_corrected.ipynb` as reusable
library functions and a CLI counterpart.

Features
- Discover raw cases under a dataset root (default expects GIST-like structure).
- Prepare SAM-Med3D-ready folders: imagesTr/labelsTr with binary labels, aligned geometry.
- Create a validation split by copying a portion of cases into imagesVal/labelsVal.

Notes
- This module purposefully avoids heavy dependencies; it only requires numpy and SimpleITK.
- Multiple lesion images/masks are detected and merged. For images, "maximum intensity" merge is used.
  For masks, a binary union (logical OR) is used.
- By default we follow the order from the notebook: first prepare imagesTr/labelsTr, then copy a
  portion to imagesVal/labelsVal for validation.
"""

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


# ------------------------------
# Paths and configuration
# ------------------------------

@dataclass
class Sam3DPaths:
    """Container for the target SAM-Med3D directory structure for one dataset."""

    sam3d_root: Path
    category: str = "gist"
    ct_name: str = "ct_GIST"

    @property
    def train_root(self) -> Path:
        return self.sam3d_root / "data" / "train" / self.category / self.ct_name

    @property
    def val_root(self) -> Path:
        return self.sam3d_root / "data" / "validation" / self.category / self.ct_name

    @property
    def images_tr(self) -> Path:
        return self.train_root / "imagesTr"

    @property
    def labels_tr(self) -> Path:
        return self.train_root / "labelsTr"

    @property
    def images_val(self) -> Path:
        return self.val_root / "imagesVal"

    @property
    def labels_val(self) -> Path:
        return self.val_root / "labelsVal"

    def ensure(self) -> None:
        for p in [self.images_tr, self.labels_tr, self.images_val, self.labels_val]:
            p.mkdir(parents=True, exist_ok=True)


def find_default_sam3d_root(start: Optional[Path] = None) -> Path:
    """Try to auto-detect the SAM-Med3D code root from current working directory upwards.

    Returns the inner repo path like: <PROJECT_ROOT>/SAM-Med3D-main/SAM-Med3D-main
    If not found, fallback to current working directory.
    """
    start = (start or Path.cwd()).resolve()
    for cand in [start, *start.parents]:
        inner = cand / "SAM-Med3D-main" / "SAM-Med3D-main"
        if inner.is_dir():
            return inner
    return start


# ------------------------------
# Case discovery
# ------------------------------

def find_case_dirs(dataset_root: Path, case_glob: Optional[str] = None) -> List[Path]:
    """Find per-case NIFTI folders.

    Default behavior mirrors the GIST layout: <root>/GIST-*_CT/1/NIFTI
    - If `case_glob` is provided, we use that (relative to dataset_root).
    - Otherwise we try the GIST pattern and fallback to scanning for directories that contain
      image/segmentation files.
    """
    dataset_root = Path(dataset_root)

    if case_glob:
        matches = sorted(dataset_root.glob(case_glob))
        return [p for p in matches if p.is_dir()]

    # 1) GIST-like pattern first
    candid = sorted(dataset_root.glob("GIST-*_CT/1/NIFTI"))
    if candid:
        return candid

    # 2) Fallback: scan shallowly for subdirs named 'NIFTI' with expected files
    #    This is conservative to avoid deep recursion on huge trees
    candid = []
    for p in dataset_root.glob("*/*/NIFTI"):
        if p.is_dir() and _has_expected_files(p):
            candid.append(p)
    if candid:
        return sorted(candid)

    # 3) Last resort: recursive search for 'NIFTI' dirs (may be costly)
    for p in dataset_root.rglob("NIFTI"):
        if p.is_dir() and _has_expected_files(p):
            candid.append(p)
    return sorted(candid)


def _has_expected_files(nifti_dir: Path) -> bool:
    """Heuristically check if a NIFTI directory contains expected image/seg files."""
    if (nifti_dir / "image.nii.gz").exists():
        return True
    if list(nifti_dir.glob("image_lesion_*.nii.gz")):
        return True
    # even if images are not found, maybe we still have labels; allow it
    if (nifti_dir / "segmentation.nii.gz").exists():
        return True
    if list(nifti_dir.glob("segmentation_lesion_*.nii.gz")):
        return True
    if list(nifti_dir.glob("segmentation_*.nii.gz")):
        return True
    return False


# ------------------------------
# I/O helpers and merging
# ------------------------------

def read_image(path: Path) -> sitk.Image:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return sitk.ReadImage(str(path))


def resample_like(img: sitk.Image, reference: sitk.Image, is_label: bool = False) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interp)
    resampler.SetTransform(sitk.Transform())
    resampler.SetOutputPixelType(img.GetPixelID())
    return resampler.Execute(img)


def to_binary_mask(label_img: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(label_img)
    bin_arr = (arr > 0).astype("uint8")
    out = sitk.GetImageFromArray(bin_arr)
    out.CopyInformation(label_img)
    return out


def find_image_files(nifti_dir: Path) -> List[Path]:
    """Find all image files in the directory (image.nii.gz or image_lesion_*.nii.gz)."""
    nifti_dir = Path(nifti_dir)
    main_image = nifti_dir / "image.nii.gz"
    if main_image.exists():
        return [main_image]
    lesion_images = sorted(nifti_dir.glob("image_lesion_*.nii.gz"))
    return lesion_images


def find_segmentation_files(nifti_dir: Path) -> List[Path]:
    """Find all segmentation files in the directory."""
    nifti_dir = Path(nifti_dir)
    main_seg = nifti_dir / "segmentation.nii.gz"
    if main_seg.exists():
        return [main_seg]
    lesion_segs = sorted(nifti_dir.glob("segmentation_lesion_*.nii.gz"))
    if not lesion_segs:
        lesion_segs = sorted(nifti_dir.glob("segmentation_*.nii.gz"))
    return lesion_segs


def merge_images(image_paths: Sequence[Path]) -> sitk.Image:
    """Merge multiple images by taking the first one as reference and max-combining others.

    If a single image is provided, it is returned as-is.
    """
    if len(image_paths) == 0:
        raise ValueError("No image paths provided")
    if len(image_paths) == 1:
        return sitk.ReadImage(str(image_paths[0]))

    merged = sitk.ReadImage(str(image_paths[0]))
    merged_arr = sitk.GetArrayFromImage(merged)

    for img_path in image_paths[1:]:
        img = sitk.ReadImage(str(img_path))
        if (
            img.GetSize() != merged.GetSize()
            or img.GetSpacing() != merged.GetSpacing()
            or img.GetDirection() != merged.GetDirection()
            or img.GetOrigin() != merged.GetOrigin()
        ):
            img = resample_like(img, reference=merged, is_label=False)
        img_arr = sitk.GetArrayFromImage(img)
        merged_arr = np.maximum(merged_arr, img_arr)

    out = sitk.GetImageFromArray(merged_arr)
    out.CopyInformation(merged)
    return out


def merge_segmentations(seg_paths: Sequence[Path]) -> sitk.Image:
    """Merge multiple segmentations by union (OR operation)."""
    if len(seg_paths) == 0:
        raise ValueError("No segmentation paths provided")
    if len(seg_paths) == 1:
        return sitk.ReadImage(str(seg_paths[0]))

    merged = sitk.ReadImage(str(seg_paths[0]))
    merged_arr = sitk.GetArrayFromImage(merged) > 0

    for seg_path in seg_paths[1:]:
        seg = sitk.ReadImage(str(seg_path))
        if (
            seg.GetSize() != merged.GetSize()
            or seg.GetSpacing() != merged.GetSpacing()
            or seg.GetDirection() != merged.GetDirection()
            or seg.GetOrigin() != merged.GetOrigin()
        ):
            seg = resample_like(seg, reference=merged, is_label=True)
        seg_arr = sitk.GetArrayFromImage(seg) > 0
        merged_arr |= seg_arr

    out = sitk.GetImageFromArray(merged_arr.astype("uint8"))
    out.CopyInformation(merged)
    return out


# ------------------------------
# Main operations
# ------------------------------

def prepare_for_sam3d(
    dataset_root: Path,
    sam3d_root: Path,
    category: str = "gist",
    ct_name: str = "ct_GIST",
    case_glob: Optional[str] = None,
    max_cases: Optional[int] = None,
) -> Tuple[int, Sam3DPaths]:
    """Prepare dataset into SAM-Med3D folder layout under `sam3d_root`.

    - Discovers cases under `dataset_root`.
    - For each case, merges multiple images/segmentations when present.
    - Aligns image geometry to label geometry; converts labels to binary.
    - Writes outputs into `data/train/<category>/<ct_name>/{imagesTr,labelsTr}` as `<case_id>.nii.gz`.

    Returns (n_prepared, paths).
    """
    dataset_root = Path(dataset_root)
    sam3d_root = Path(sam3d_root)

    paths = Sam3DPaths(sam3d_root=sam3d_root, category=category, ct_name=ct_name)
    paths.ensure()

    nifti_dirs = find_case_dirs(dataset_root, case_glob=case_glob)
    if max_cases is not None:
        nifti_dirs = nifti_dirs[:max_cases]

    prepared = 0
    for case_dir in nifti_dirs:
        # Try to derive a case_id from directory layout
        # Expectation: <case>/1/NIFTI -> case_id = <case> folder name
        try:
            case_id = case_dir.parent.parent.name
        except Exception:
            case_id = case_dir.name  # fallback

        try:
            image_files = find_image_files(case_dir)
            seg_files = find_segmentation_files(case_dir)

            if not image_files:
                print(f"[SKIP] {case_id}: No image files found in {case_dir}")
                continue
            if not seg_files:
                print(f"[SKIP] {case_id}: No segmentation files found in {case_dir}")
                continue

            img = merge_images(image_files)
            lbl = merge_segmentations(seg_files)
        except Exception as e:
            print(f"[SKIP] {case_id}: {e}")
            continue

        # Align image geometry to label geometry
        if (
            img.GetSize() != lbl.GetSize()
            or img.GetSpacing() != lbl.GetSpacing()
            or img.GetDirection() != lbl.GetDirection()
            or img.GetOrigin() != lbl.GetOrigin()
        ):
            img = resample_like(img, reference=lbl, is_label=False)

        lbl_bin = to_binary_mask(lbl)
        out_image = paths.images_tr / f"{case_id}.nii.gz"
        out_label = paths.labels_tr / f"{case_id}.nii.gz"
        sitk.WriteImage(img, str(out_image))
        sitk.WriteImage(lbl_bin, str(out_label))

        prepared += 1
        if prepared % 25 == 0:
            print(f"Prepared {prepared} cases ...")

    print(f"Done. Prepared {prepared} cases to {paths.train_root}")
    return prepared, paths


def split_validation(
    paths: Sam3DPaths,
    split_ratio: float = 0.8,
    seed: int = 2025,
    copy: bool = True,
) -> Tuple[int, int]:
    """Create a validation split by copying/moving a portion of imagesTr/labelsTr into imagesVal/labelsVal.

    By default, we COPY (non-destructive) to keep the full training set intact under imagesTr/labelsTr
    and mirror the validation subset under imagesVal/labelsVal, which matches the notebook behavior.

    Returns (n_train, n_val) based on the split.
    """
    paths.ensure()

    images = sorted(paths.images_tr.glob("*.nii.gz"))
    labels = sorted(paths.labels_tr.glob("*.nii.gz"))
    if len(images) != len(labels):
        raise RuntimeError(
            f"imagesTr and labelsTr count mismatch: {len(images)} vs {len(labels)}"
        )

    idxs = list(range(len(images)))
    random.Random(seed).shuffle(idxs)
    split = int(len(images) * split_ratio)
    train_idxs, val_idxs = idxs[:split], idxs[split:]

    op = shutil.copy2 if copy else shutil.move

    for i in val_idxs:
        src_i = images[i]
        src_l = labels[i]
        op(src_i, paths.images_val / src_i.name)
        op(src_l, paths.labels_val / src_l.name)

    print(
        f"Validation set {'copied' if copy else 'moved'} to: {paths.val_root} | Train: {len(train_idxs)} | Val: {len(val_idxs)}"
    )
    return len(train_idxs), len(val_idxs)
