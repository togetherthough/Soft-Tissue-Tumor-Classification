# Troubleshooting

Use this guide when fine-tuning SAM-Med3D or running PFN experiments.

- **CPU vs GPU checkpoints**
  - Error: `Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`
  - Cause: Checkpoint was saved on GPU, loaded on CPU.
  - Fix: Load with CPU map_location and/or use a CPU-saved checkpoint.
    ```python
    import torch
    ckpt = torch.load(".../sam_med3d_turbo.pth", map_location="cpu")
    torch.save(ckpt, ".../sam_med3d_turbo_cpu.pth")
    # then pass ..._cpu.pth to training
    ```

- **OpenMP duplicate runtime (libomp vs libiomp)**
  - Error at process start mentioning `libomp.dll` and `libiomp5md.dll`.
  - Fix: set env var before spawning training:
    ```python
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    ```

- **DataLoader workers too high on CPU**
  - Warning: `This DataLoader will create 24 worker processes...`
  - Symptom: Slow or stalled CPU runs.
  - Fix: reduce workers via training args, e.g. `--num_workers 4` or `8`.

- **Autocast warning on CPU**
  - Message: `torch.cuda.amp.GradScaler is enabled, but CUDA is not available. Disabling.`
  - This is harmless; training continues. Autocast is disabled on CPU.

- **No checkpoint saved (crash during epoch)**
  - Common causes:
    - Loss expects float targets: ensure masks are float (`gt3D.float()`).
    - Checkpoint device mismatch (see CPU vs GPU checkpoints above).
    - Excessive workers on CPU (reduce workers).
  - Where to look:
    - Work dir: `notebooks/finetuned_pfn_results/<dataset>_finetune_workdir/<category>_<ct_name>_ft/`
    - Files: `sam_model_latest.pth`, `sam_model_dice_best.pth`, `Loss.png`, `Dice.png`

- **Where are logs?**
  - If launched via the Python wrapper, stdout is streamed to the notebook.
  - Native SAM-Med3D writes under `<work_dir>/<task_name>/`.

- **Path assumptions**
  - SAM-Med3D lives at `./sam-med3d/` relative to project root.
  - Prepared data lives under that repo: `data/train/<category>/<ct_name>/{imagesTr,labelsTr}`.

If issues persist, open `sam-med3d/train.py` (if it exists) and verify:
- `device_config()` respects CPU when CUDA is unavailable.
- `init_checkpoint()` loads with `map_location='cpu'` as fallback.
- Training loop casts `gt3D.float()` for Dice losses with `sigmoid=True`.
