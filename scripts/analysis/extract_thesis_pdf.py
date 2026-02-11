import re
from pathlib import Path

import fitz  # PyMuPDF


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return (text[:max_len].rstrip("-")) or "untitled"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = repo_root / "thesis_2nd_draft.pdf"
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = repo_root / "docs" / "_pdf_extract" / pdf_path.stem
    pages_dir = out_dir / "pages"
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))

    md_lines = []
    md_lines.append(f"# Extract: {pdf_path.name}\n")
    md_lines.append(f"- Pages: {doc.page_count}")
    md_lines.append(f"- Output: {out_dir.as_posix()}\n")

    # Render pages + extract text
    zoom = 2.0  # ~144 DPI equivalent
    mat = fitz.Matrix(zoom, zoom)

    for i in range(doc.page_count):
        page = doc.load_page(i)
        page_num = i + 1

        text = page.get_text("text") or ""
        # Normalize a bit for markdown
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        md_lines.append(f"\n## Page {page_num}\n")
        md_lines.append(text if text else "(no extractable text)")

        # Render page image
        pix = page.get_pixmap(matrix=mat, alpha=False)
        page_png = pages_dir / f"page_{page_num:03d}.png"
        pix.save(str(page_png))

        # Extract embedded images (if any)
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base = doc.extract_image(xref)
            ext = base.get("ext", "bin")
            img_bytes = base.get("image")
            if not img_bytes:
                continue
            img_name = images_dir / f"page_{page_num:03d}_img_{img_index:02d}.{ext}"
            img_name.write_bytes(img_bytes)

    md_path = out_dir / f"{pdf_path.stem}_text.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    index_lines = []
    index_lines.append(f"# Figure/page index for {pdf_path.name}\n")
    index_lines.append("## Rendered pages")
    index_lines.append(f"- Folder: {pages_dir.as_posix()}")
    index_lines.append("- Files: page_###.png\n")

    index_lines.append("## Extracted embedded images")
    index_lines.append(f"- Folder: {images_dir.as_posix()}")
    extracted = sorted(images_dir.glob("*"))
    if extracted:
        for p in extracted:
            index_lines.append(f"- {p.name}")
    else:
        index_lines.append("- (none found — many plots are vector drawings and won’t appear as embedded raster images)")

    index_path = out_dir / "INDEX.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {md_path}")
    print(f"Wrote: {index_path}")
    print(f"Rendered pages: {pages_dir}")
    print(f"Extracted images: {images_dir}")


if __name__ == "__main__":
    main()
