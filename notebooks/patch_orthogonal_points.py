import json
from pathlib import Path

nb_path = Path('SAM_Visualization_ORTHOGONAL_FINAL.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code' and any('def plot_orthogonal_views' in s for s in cell.get('source', [])):
        cell_idx = i
        break

if cell_idx is None:
    raise SystemExit('plot_orthogonal_views cell not found')

new_source = r'''
import numpy as np
import matplotlib.pyplot as plt

def plot_orthogonal_views(image_vol, seg_vol, gt_mask_vol=None, title="", n_slices=3, modality="CT", points=None):
    D, H, W = image_vol.shape

    # display window
    fg = image_vol[image_vol > (image_vol.min() + 0.05 * (image_vol.max() - image_vol.min()))]
    if fg.size > 100:
        if "MR" in modality.upper():
            vmin, vmax = np.percentile(fg, [1, 99])
        else:
            mu, sd = fg.mean(), fg.std()
            vmin, vmax = mu - 2*sd, mu + 2*sd
            vmin = max(vmin, fg.min()); vmax = min(vmax, fg.max())
    else:
        vmin, vmax = image_vol.min(), image_vol.max()

    # choose 3D points (Z,Y,X)
    sel_points = []
    if points is not None and len(points) > 0:
        sel_points = [tuple(int(p[k]) for k in range(3)) for p in points][:n_slices]
    elif gt_mask_vol is not None and gt_mask_vol.sum() > 0:
        coords = np.argwhere(gt_mask_vol > 0)  # (N,3) -> (z,y,x)
        c = coords.astype(float)
        center = c.mean(axis=0, keepdims=True)
        cc = c - center
        # PCA principal axis
        try:
            _, _, vt = np.linalg.svd(cc, full_matrices=False)
            axis = vt[0]
            proj = (cc @ axis)
            qs = np.linspace(0.1, 0.9, max(3, n_slices))
            idxs = [int(np.argmin(np.abs(proj - np.quantile(proj, q)))) for q in qs]
            for idx in idxs:
                z, y, x = coords[idx]
                sel_points.append((int(z), int(y), int(x)))
        except np.linalg.LinAlgError:
            pass
        # fallback to center if PCA failed
        if not sel_points:
            cz, cy, cx = center[0].astype(int)
            sel_points = [(int(cz), int(cy), int(cx))]
        # ensure unique and clamp
        uniq = []
        seen = set()
        for p in sel_points:
            z = int(np.clip(p[0], 0, D-1)); y = int(np.clip(p[1], 0, H-1)); x = int(np.clip(p[2], 0, W-1))
            if (z,y,x) not in seen:
                uniq.append((z,y,x)); seen.add((z,y,x))
        sel_points = uniq[:n_slices]
        # if fewer than requested, pad with center-line variations along Z
        while len(sel_points) < n_slices:
            cz = int(np.median([p[0] for p in sel_points])) if sel_points else D//2
            cy = int(np.median([p[1] for p in sel_points])) if sel_points else H//2
            cx = int(np.median([p[2] for p in sel_points])) if sel_points else W//2
            for dz in (-D//6, 0, D//6):
                z = int(np.clip(cz + dz, 0, D-1))
                p = (z, cy, cx)
                if p not in sel_points:
                    sel_points.append(p)
                if len(sel_points) == n_slices:
                    break
    else:
        # no mask -> center-line variations
        cz, cy, cx = D//2, H//2, W//2
        zvals = np.linspace(max(0, cz - D//6), min(D-1, cz + D//6), n_slices).astype(int)
        sel_points = [(int(z), cy, cx) for z in zvals]

    print('Selected 3D points (Z,Y,X):', sel_points)

    # plotting with crosshairs to confirm voxel alignment
    n_cols = 3 if gt_mask_vol is not None else 2
    fig = plt.figure(figsize=(5*3, 4*len(sel_points)))

    for i, (z, y, x) in enumerate(sel_points):
        # AXIAL (rows=H, cols=W)
        img_a = image_vol[z, :, :]
        seg_a = seg_vol[z, :, :]
        gt_a = gt_mask_vol[z, :, :] if gt_mask_vol is not None else None
        base = (i*3) * n_cols
        ax = fig.add_subplot(len(sel_points)*3, n_cols, base + 1)
        ax.imshow(img_a, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axhline(y, color='y', alpha=0.6, lw=0.6)
        ax.axvline(x, color='y', alpha=0.6, lw=0.6)
        ax.set_title(f'AXIAL (Z={z})'); ax.axis('off')
        ax = fig.add_subplot(len(sel_points)*3, n_cols, base + 2)
        ax.imshow(img_a, cmap='gray', vmin=vmin, vmax=vmax)
        ax.imshow(np.ma.masked_where(seg_a == 0, seg_a), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        ax.axhline(y, color='y', alpha=0.6, lw=0.6)
        ax.axvline(x, color='y', alpha=0.6, lw=0.6)
        ax.set_title(f'AXIAL SAM (Z={z})'); ax.axis('off')
        if gt_mask_vol is not None:
            ax = fig.add_subplot(len(sel_points)*3, n_cols, base + 3)
            ax.imshow(img_a, cmap='gray', vmin=vmin, vmax=vmax)
            ax.imshow(np.ma.masked_where(gt_a == 0, gt_a), cmap='Greens', alpha=0.5, vmin=0, vmax=1)
            ax.axhline(y, color='y', alpha=0.6, lw=0.6)
            ax.axvline(x, color='y', alpha=0.6, lw=0.6)
            ax.set_title(f'AXIAL GT (Z={z})'); ax.axis('off')

        # CORONAL (rows=D, cols=W) -> crosshair at (row=z, col=x)
        img_c = image_vol[:, y, :]
        seg_c = seg_vol[:, y, :]
        gt_c = gt_mask_vol[:, y, :] if gt_mask_vol is not None else None
        base = (i*3 + 1) * n_cols
        ax = fig.add_subplot(len(sel_points)*3, n_cols, base + 1)
        ax.imshow(img_c, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axhline(z, color='y', alpha=0.6, lw=0.6)
        ax.axvline(x, color='y', alpha=0.6, lw=0.6)
        ax.set_title(f'CORONAL (Y={y})'); ax.axis('off')
        ax = fig.add_subplot(len(sel_points)*3, n_cols, base + 2)
        ax.imshow(img_c, cmap='gray', vmin=vmin, vmax=vmax)
        ax.imshow(np.ma.masked_where(seg_c == 0, seg_c), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        ax.axhline(z, color='y', alpha=0.6, lw=0.6)
        ax.axvline(x, color='y', alpha=0.6, lw=0.6)
        ax.set_title(f'CORONAL SAM (Y={y})'); ax.axis('off')
        if gt_mask_vol is not None:
            ax = fig.add_subplot(len(sel_points)*3, n_cols, base + 3)
            ax.imshow(img_c, cmap='gray', vmin=vmin, vmax=vmax)
            ax.imshow(np.ma.masked_where(gt_c == 0, gt_c), cmap='Greens', alpha=0.5, vmin=0, vmax=1)
            ax.axhline(z, color='y', alpha=0.6, lw=0.6)
            ax.axvline(x, color='y', alpha=0.6, lw=0.6)
            ax.set_title(f'CORONAL GT (Y={y})'); ax.axis('off')

        # SAGITTAL (rows=D, cols=H) -> crosshair at (row=z, col=y)
        img_s = image_vol[:, :, x]
        seg_s = seg_vol[:, :, x]
        gt_s = gt_mask_vol[:, :, x] if gt_mask_vol is not None else None
        base = (i*3 + 2) * n_cols
        ax = fig.add_subplot(len(sel_points)*3, n_cols, base + 1)
        ax.imshow(img_s, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axhline(z, color='y', alpha=0.6, lw=0.6)
        ax.axvline(y, color='y', alpha=0.6, lw=0.6)
        ax.set_title(f'SAGITTAL (X={x})'); ax.axis('off')
        ax = fig.add_subplot(len(sel_points)*3, n_cols, base + 2)
        ax.imshow(img_s, cmap='gray', vmin=vmin, vmax=vmax)
        ax.imshow(np.ma.masked_where(seg_s == 0, seg_s), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        ax.axhline(z, color='y', alpha=0.6, lw=0.6)
        ax.axvline(y, color='y', alpha=0.6, lw=0.6)
        ax.set_title(f'SAGITTAL SAM (X={x})'); ax.axis('off')
        if gt_mask_vol is not None:
            ax = fig.add_subplot(len(sel_points)*3, n_cols, base + 3)
            ax.imshow(img_s, cmap='gray', vmin=vmin, vmax=vmax)
            ax.imshow(np.ma.masked_where(gt_s == 0, gt_s), cmap='Greens', alpha=0.5, vmin=0, vmax=1)
            ax.axhline(z, color='y', alpha=0.6, lw=0.6)
            ax.axvline(y, color='y', alpha=0.6, lw=0.6)
            ax.set_title(f'SAGITTAL GT (X={x})'); ax.axis('off')

    fig.suptitle(f"{title}\n[Orthogonal Views - Distinct 3D Points | {modality}]", fontsize=14)
    plt.tight_layout()
    return fig
'''

nb['cells'][cell_idx]['source'] = [line + '\n' for line in new_source.strip('\n').split('\n')]

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('Patched plot_orthogonal_views with distinct 3D points and crosshairs')
