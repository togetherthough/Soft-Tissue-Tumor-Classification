from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    fontsize=8.6,
    fc="#FFFFFF",
    ec="#2F2F2F",
    lw=0.9,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111111",
        wrap=True,
    )
    return patch


def _arrow(ax, x0, y0, x1, y1, *, color="#2F2F2F"):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=0.9,
        color=color,
    )
    ax.add_patch(arr)
    return arr


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Two-column friendly: tall, narrow figure (publication style).
    fig = plt.figure(figsize=(3.4, 5.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Global styling (LaTeX-like, clean)
    mpl.rcParams["font.family"] = "DejaVu Serif"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    # Palette (subtle, print-friendly)
    border = "#2F2F2F"
    soft = "#F5F6F8"
    accent = "#DDE7F7"
    accent2 = "#E7F2E7"

    # Title
    ax.text(
        0.08,
        0.965,
        "Feature Extraction Pipeline",
        ha="left",
        va="top",
        fontsize=10.2,
        color=border,
        fontweight="bold",
    )
    ax.text(
        0.08,
        0.938,
        "ROI-cropped volumes → SAM-Med3D embeddings → pooled features → tabular classifier",
        ha="left",
        va="top",
        fontsize=7.8,
        color=border,
    )

    w = 0.84
    h = 0.092
    x = 0.08

    ys = [0.84, 0.72, 0.60, 0.48, 0.36, 0.24, 0.12]

    b1 = _box(ax, x, ys[0], w, h, "Input volume (CT/MRI, 3D)", fc=soft, ec=border)
    b2 = _box(ax, x, ys[1], w, h, "ROI extraction (lesion-centered crop + pad)", fc=accent2, ec=border)
    b3 = _box(ax, x, ys[2], w, h, "SAM-Med3D image encoder (frozen)", fc=accent, ec=border)
    b4 = _box(ax, x, ys[3], w, h, "Dense embedding map  (C × d × h × w)", fc=soft, ec=border)
    b5 = _box(ax, x, ys[4], w, h, "Global average pooling", fc=soft, ec=border)
    b6 = _box(ax, x, ys[5], w, h, "Study-level feature vector  (C,)", fc=soft, ec=border)
    b7 = _box(ax, x, ys[6], w, h, "Downstream classifier  (TabPFN / LoCalPFN)", fc=soft, ec=border)

    boxes = [b1, b2, b3, b4, b5, b6, b7]
    for a, b in zip(boxes[:-1], boxes[1:]):
        x0 = a.get_x() + a.get_width() / 2
        y0 = a.get_y()
        x1 = b.get_x() + b.get_width() / 2
        y1 = b.get_y() + b.get_height()
        _arrow(ax, x0, y0, x1, y1, color=border)

    # Side callouts (minimal)
    ax.text(0.08, 0.565, "Pretrained", ha="left", va="center", fontsize=7.6, color=border)
    ax.text(0.08, 0.445, "Embeddings", ha="left", va="center", fontsize=7.6, color=border)
    ax.text(0.08, 0.205, "Features", ha="left", va="center", fontsize=7.6, color=border)

    base = out_dir / "feature_extraction_pipeline"
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print("Wrote:", base.with_suffix(".png"))
    print("Wrote:", base.with_suffix(".pdf"))
    print("Wrote:", base.with_suffix(".svg"))


if __name__ == "__main__":
    main()
