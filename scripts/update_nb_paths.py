from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

# Replace runs path strings in code and markdown cells (do not touch outputs)
REPLACEMENTS: Dict[str, str] = {
    # Windows-like paths
    r"notebooks\\vision3d_runs": r"notebooks\\baselines",
    # POSIX-like paths
    "notebooks/vision3d_runs": "notebooks/baselines",
    # Any remaining bare token
    "vision3d_runs": "baselines",
}


def update_notebook(nb_path: Path) -> int:
    with nb_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed_cells = 0
    for cell in data.get("cells", []):
        if cell.get("cell_type") not in {"code", "markdown"}:
            continue
        src = cell.get("source", [])
        new_src = []
        cell_changed = False
        for line in src:
            new_line = line
            for old, new in REPLACEMENTS.items():
                if old in new_line:
                    new_line = new_line.replace(old, new)
            if new_line != line:
                cell_changed = True
            new_src.append(new_line)
        if cell_changed:
            cell["source"] = new_src
            changed_cells += 1

    if changed_cells:
        bak = nb_path.with_suffix(f".ipynb.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        bak.write_text(nb_path.read_text(encoding="utf-8"), encoding="utf-8")
        with nb_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.write("\n")
    return changed_cells


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/update_nb_paths.py <notebook_path.ipynb>")
        return 2
    nb_path = Path(argv[1])
    if not nb_path.exists():
        print(f"Notebook not found: {nb_path}")
        return 1
    changed = update_notebook(nb_path)
    print(f"Updated {changed} cells in {nb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
