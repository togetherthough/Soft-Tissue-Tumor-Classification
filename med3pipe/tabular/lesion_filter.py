from __future__ import annotations

"""
med3pipe.tabular.lesion_filter

Utilities for filtering cases based on lesion size metrics from preprocessing analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd


@dataclass
class LesionSizeFilter:
    """Configuration for filtering cases by lesion size metrics.
    
    All thresholds are minimum values (inclusive). Set to None to disable that filter.
    
    Attributes:
        min_voxels: Minimum preprocessed voxel count (e.g., 500)
        min_dimension: Minimum bounding box dimension (e.g., 5)
        min_density: Minimum lesion density (voxels / bbox volume, e.g., 0.3)
        lesion_csv_path: Path to lesion_size_analysis.csv (auto-detected if None)
    """
    min_voxels: Optional[int] = None
    min_dimension: Optional[int] = None
    min_density: Optional[float] = None
    lesion_csv_path: Optional[Path] = None
    
    def __post_init__(self):
        """Auto-detect lesion CSV if not provided."""
        if self.lesion_csv_path is None:
            # Try to find repo root (go up from this module)
            module_dir = Path(__file__).parent.parent.parent.resolve()  # med3pipe/tabular/lesion_filter.py -> repo root
            
            # Try to find it in standard locations (tracked location first)
            candidates = [
                # Relative to repo root (works on cluster)
                module_dir / "data" / "lesion_size_analysis.csv",
                module_dir / "results" / "lesion_preprocessing" / "lesion_size_analysis.csv",
                module_dir / "results" / "lesion_size_analysis.csv",
                # Relative to cwd (fallback)
                Path.cwd() / "data" / "lesion_size_analysis.csv",
                Path.cwd() / "results" / "lesion_preprocessing" / "lesion_size_analysis.csv",
                Path.cwd() / "results" / "lesion_size_analysis.csv",
            ]
            for cand in candidates:
                if cand.exists():
                    self.lesion_csv_path = cand
                    break
    
    def is_enabled(self) -> bool:
        """Check if any filtering is enabled."""
        return (
            self.min_voxels is not None
            or self.min_dimension is not None
            or self.min_density is not None
        )
    
    def get_valid_cases(self) -> Set[str]:
        """Get set of case IDs that pass all enabled filters.
        
        Returns:
            Set of case IDs (without dataset prefix or file extension)
            
        Raises:
            FileNotFoundError: If lesion CSV not found
            ValueError: If filtering enabled but CSV has no data
        """
        if not self.is_enabled():
            return set()  # No filtering, allow all cases
        
        if self.lesion_csv_path is None or not Path(self.lesion_csv_path).exists():
            raise FileNotFoundError(
                f"Lesion size analysis CSV not found at {self.lesion_csv_path}. "
                f"Run scripts/analyze_lesion_sizes.py first."
            )
        
        df = pd.read_csv(self.lesion_csv_path)
        
        if len(df) == 0:
            raise ValueError(f"Lesion CSV {self.lesion_csv_path} is empty")
        
        # Apply filters
        mask = pd.Series([True] * len(df))
        
        if self.min_voxels is not None:
            mask &= df["preproc_voxels"] >= self.min_voxels
        
        if self.min_dimension is not None:
            mask &= df["min_dimension"] >= self.min_dimension
        
        if self.min_density is not None:
            mask &= df["density"] >= self.min_density
        
        filtered_df = df[mask]
        
        # Extract case IDs (column is "case")
        case_ids = set(filtered_df["case"].astype(str))
        
        return case_ids
    
    def print_filter_summary(self) -> None:
        """Print a summary of the filtering configuration and results."""
        if not self.is_enabled():
            print("[INFO] Lesion size filtering: DISABLED")
            return
        
        print("\n" + "="*70)
        print("LESION SIZE FILTERING")
        print("="*70)
        
        criteria = []
        if self.min_voxels is not None:
            criteria.append(f"  • Preprocessed voxels >= {self.min_voxels}")
        if self.min_dimension is not None:
            criteria.append(f"  • Minimum dimension >= {self.min_dimension}")
        if self.min_density is not None:
            criteria.append(f"  • Lesion density >= {self.min_density:.2f}")
        
        print("Filtering criteria:")
        print("\n".join(criteria))
        
        try:
            valid_cases = self.get_valid_cases()
            
            # Load full dataset to compute statistics
            if self.lesion_csv_path and Path(self.lesion_csv_path).exists():
                df = pd.read_csv(self.lesion_csv_path)
                total_cases = len(df)
                viable_cases = len(valid_cases)
                pct = (viable_cases / total_cases * 100) if total_cases > 0 else 0
                
                print(f"\nResults:")
                print(f"  • Total cases: {total_cases}")
                print(f"  • Viable cases: {viable_cases} ({pct:.1f}%)")
                print(f"  • Filtered out: {total_cases - viable_cases}")
                
                # Per-dataset breakdown
                df_viable = df[df["case"].isin(valid_cases)]
                print(f"\nPer-dataset breakdown:")
                for dataset in sorted(df["dataset"].unique()):
                    ds_total = (df["dataset"] == dataset).sum()
                    ds_viable = (df_viable["dataset"] == dataset).sum()
                    ds_pct = (ds_viable / ds_total * 100) if ds_total > 0 else 0
                    print(f"  • {dataset}: {ds_viable}/{ds_total} ({ds_pct:.1f}%)")
            else:
                print(f"\nViable cases: {len(valid_cases)}")
        except Exception as e:
            print(f"\n⚠️  Error loading lesion data: {e}")
        
        print("="*70 + "\n")


def load_lesion_filter_from_config(config: Dict) -> Optional[LesionSizeFilter]:
    """Load lesion filter from a configuration dict.
    
    Args:
        config: Dictionary with keys 'min_voxels', 'min_dimension', 'min_density'
        
    Returns:
        LesionSizeFilter if any filter is specified, None otherwise
    """
    min_voxels = config.get("min_voxels")
    min_dimension = config.get("min_dimension")
    min_density = config.get("min_density")
    lesion_csv_path = config.get("lesion_csv_path")
    
    if min_voxels is None and min_dimension is None and min_density is None:
        return None
    
    return LesionSizeFilter(
        min_voxels=min_voxels,
        min_dimension=min_dimension,
        min_density=min_density,
        lesion_csv_path=Path(lesion_csv_path) if lesion_csv_path else None,
    )
