"""Debug SAM-Med3D preprocessing for representative cases."""

from importlib.machinery import SourceFileLoader
from pathlib import Path
import numpy as np

# Load helpers from compute_sam_dice_scores without executing main
script_path = Path(__file__).parent / "compute_sam_dice_scores.py"
loader = SourceFileLoader("sam_dice_script", str(script_path))
mod = loader.load_module()

project_root = Path(__file__).parent.parent
config_path = project_root / "configs" / "datasets_analysis.yaml"
print(f"Loading datasets from {config_path}")

datasets = mod.discover_cases_from_config(config_path, project_root)
print("Datasets discovered:", {k: len(v['cases']) for k, v in datasets.items()})

for dataset_name, info in datasets.items():
    if not info['cases']:
        print(f"\n[SKIP] {dataset_name}: no cases")
        continue
    case = info['cases'][0]
    modality = info.get('modality', 'CT')

    print("\n" + "=" * 80)
    print(f"Dataset: {dataset_name} | Case: {case['case_id']} | Modality: {modality}")

    image_sitk = mod.merge_images(case['image_files'])
    mask_sitk = mod.merge_segmentations(case['segmentation_files'])

    img_arr = np.array(mod.sitk.GetArrayFromImage(image_sitk))
    mask_arr = np.array(mod.sitk.GetArrayFromImage(mask_sitk))

    print("Raw image size:", image_sitk.GetSize(), "spacing:", image_sitk.GetSpacing())
    print(
        "Raw image stats: min",
        float(img_arr.min()),
        "max",
        float(img_arr.max()),
        "mean",
        float(img_arr.mean()),
    )
    print(
        "Raw mask size:", mask_sitk.GetSize(), "voxels:", int(mask_arr.sum()),
    )

    processed_img = mod.load_volume_for_sam(
        image_sitk,
        img_size=128,
        modality=modality,
    )
    processed_mask = mod.load_mask_for_sam(mask_sitk, img_size=128)

    print(
        "Processed image shape:",
        tuple(processed_img.shape),
        "min",
        float(processed_img.min()),
        "max",
        float(processed_img.max()),
    )
    print(
        "Processed mask shape:",
        processed_mask.shape,
        "voxels:",
        int(processed_mask.sum()),
    )
