import nbformat
from pathlib import Path
import textwrap


path = Path(r"c:\\Users\\cahel\\Desktop\\Med3Tab-PFN\\notebooks\\visualization\\Raw_Image_Display.ipynb")
nb = nbformat.read(path.open(encoding="utf-8"), as_version=4)

cell_updates = {
    "find_images": textwrap.dedent(
        """
        def find_dataset_images(dataset_config, sam3d_root, project_root):
            '''Find image/label pairs for a dataset.

            Preference order: raw project data > processed SAM-Med3D exports > cluster scratch.
            '''

            category = dataset_config.get('category')
            ct_name = dataset_config.get('ct_name', '')

            def add_pair(target_images, target_labels, seen_set, img_path, label_path=None):
                if img_path is None:
                    return
                img_path = Path(img_path)
                if not img_path.exists():
                    return
                key = img_path.resolve()
                if key in seen_set:
                    return
                seen_set.add(key)

                label_path_obj = Path(label_path) if label_path else None
                if label_path_obj and not label_path_obj.exists():
                    label_path_obj = None

                target_images.append(img_path)
                target_labels.append(label_path_obj)

            # First try raw project data (expected patient-level files)
            raw_images, raw_labels, raw_seen = [], [], set()
            data_root = project_root / 'data' / category if category else None
            if data_root and data_root.exists():
                for case_dir in sorted(data_root.glob('*')):
                    nifti_dir = case_dir / '1' / 'NIFTI'
                    if not nifti_dir.exists():
                        nifti_dir = case_dir / 'NIFTI'
                    if not nifti_dir.exists():
                        continue

                    img_files = (
                        list(nifti_dir.glob('image.nii.gz'))
                        or list(nifti_dir.glob('image_lesion_*.nii.gz'))
                        or list(nifti_dir.glob('image_*.nii.gz'))
                        or list(nifti_dir.glob('*image*.nii.gz'))
                    )
                    if not img_files:
                        continue

                    seg_files = (
                        list(nifti_dir.glob('segmentation.nii.gz'))
                        or list(nifti_dir.glob('segmentation_lesion_*.nii.gz'))
                        or list(nifti_dir.glob('segmentation_*.nii.gz'))
                        or list(nifti_dir.glob('*label*.nii.gz'))
                    )

                    add_pair(
                        raw_images,
                        raw_labels,
                        raw_seen,
                        img_files[0],
                        seg_files[0] if seg_files else None,
                    )

            if raw_images:
                return raw_images, raw_labels

            # Fallback to processed SAM-Med3D exports
            processed_images, processed_labels, processed_seen = [], [], set()
            processed_paths = []
            if category:
                processed_paths = [
                    (
                        sam3d_root / 'data' / 'train' / category / ct_name / 'imagesTr',
                        sam3d_root / 'data' / 'train' / category / ct_name / 'labelsTr',
                    ),
                    (
                        sam3d_root / 'data' / 'validation' / category / ct_name / 'imagesVal',
                        sam3d_root / 'data' / 'validation' / category / ct_name / 'labelsVal',
                    ),
                ]

            for img_dir, lbl_dir in processed_paths:
                if not img_dir.exists():
                    continue
                label_dir_exists = lbl_dir.exists()
                for img_file in sorted(img_dir.glob('*.nii.gz')):
                    label_file = None
                    if label_dir_exists:
                        candidate = lbl_dir / img_file.name
                        if candidate.exists():
                            label_file = candidate
                    add_pair(processed_images, processed_labels, processed_seen, img_file, label_file)

            if processed_images:
                return processed_images, processed_labels

            # Final fallback: cluster scratch layout
            cluster_images, cluster_labels, cluster_seen = [], [], set()
            cluster_paths = []
            if category:
                cluster_paths = [
                    (
                        Path(f"/data/scratch/r112276/{category}/{ct_name}/imagesTr"),
                        Path(f"/data/scratch/r112276/{category}/{ct_name}/labelsTr"),
                    ),
                ]

            for img_dir, lbl_dir in cluster_paths:
                if not img_dir.exists():
                    continue
                label_dir_exists = lbl_dir.exists()
                for img_file in sorted(img_dir.glob('*.nii.gz')):
                    label_file = None
                    if label_dir_exists:
                        candidate = lbl_dir / img_file.name
                        if candidate.exists():
                            label_file = candidate
                    add_pair(cluster_images, cluster_labels, cluster_seen, img_file, label_file)

            return cluster_images, cluster_labels
        """
    ).strip(),
    "process_images": textwrap.dedent(
        """
        for dataset_name, dataset_config in config['datasets'].items():
            print("\\n" + "=" * 60)
            print(f"Processing dataset: {dataset_name.upper()}")
            print("=" * 60)

            images, _ = find_dataset_images(dataset_config, sam3d_root, project_root)

            if len(images) == 0:
                print(f"  ⚠️ No images found for {dataset_name}")
                continue

            sample_count = min(n_samples_per_dataset, len(images))
            if sample_count == 0:
                print("  ⚠️ Sample count is 0; skipping dataset")
                continue

            print(f"Sampling {sample_count} image(s) for visualization")
            sample_indices = sorted(random.sample(range(len(images)), sample_count))

            modality_hint = dataset_config.get('modality', '').upper()

            for sample_rank, idx in enumerate(sample_indices, start=1):
                img_path = images[idx]

                print(f"\\n📁 Processing: {img_path.name}")
                print(f"  Sample #{sample_rank}/{sample_count}")

                try:
                    print("  Loading RAW volume (no preprocessing)...")
                    raw_volume = load_raw_volume(img_path)

                    modality = modality_hint or ("MR" if "_MR" in img_path.name.upper() else "CT")

                    coords = choose_coordinates(raw_volume, n_coords_per_sample)

                    print(f"  Plotting RAW slices in three orientations [Modality: {modality}]...")
                    fig = plot_raw_orientations(
                        raw_volume,
                        coords,
                        title=f"{dataset_name.upper()} #{sample_rank}: {img_path.stem}",
                        modality=modality,
                    )
                    plt.show()

                except Exception as e:
                    print(f"  ❌ Error processing {img_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
        """
    ).strip(),
}

remaining_updates = set(cell_updates.keys())
for cell in nb.cells:
    cell_id = cell.get("id")
    if cell_id in cell_updates:
        cell["source"] = cell_updates[cell_id]
        remaining_updates.discard(cell_id)

if remaining_updates:
    missing = ", ".join(sorted(remaining_updates))
    raise SystemExit(f"Target cell(s) not found: {missing}")

path.write_text(nbformat.writes(nb, version=4), encoding="utf-8")
