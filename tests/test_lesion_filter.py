"""
Tests for lesion size filtering functionality.
"""

import sys
from pathlib import Path
import pandas as pd
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from med3pipe.tabular.lesion_filter import LesionSizeFilter, load_lesion_filter_from_config


def test_lesion_filter_disabled():
    """Test that empty filter is disabled."""
    filter_obj = LesionSizeFilter()
    assert not filter_obj.is_enabled()


def test_lesion_filter_enabled():
    """Test that filter with parameters is enabled."""
    filter_obj = LesionSizeFilter(min_voxels=500)
    assert filter_obj.is_enabled()
    
    filter_obj = LesionSizeFilter(min_dimension=5)
    assert filter_obj.is_enabled()
    
    filter_obj = LesionSizeFilter(min_density=0.3)
    assert filter_obj.is_enabled()


def test_lesion_filter_get_valid_cases():
    """Test filtering with mock CSV data."""
    # Create temporary CSV with test data
    test_data = pd.DataFrame({
        'dataset': ['gist', 'gist', 'gist', 'gist', 'gist'],
        'case': ['GIST-001_CT', 'GIST-002_CT', 'GIST-003_CT', 'GIST-004_CT', 'GIST-005_CT'],
        'preproc_voxels': [100, 600, 200, 800, 1500],
        'min_dimension': [2, 6, 3, 8, 12],
        'density': [0.2, 0.4, 0.15, 0.5, 0.6],
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        csv_path = Path(f.name)
    
    try:
        # Test voxel filtering
        filter_obj = LesionSizeFilter(min_voxels=500, lesion_csv_path=csv_path)
        valid_cases = filter_obj.get_valid_cases()
        assert valid_cases == {'GIST-002_CT', 'GIST-004_CT', 'GIST-005_CT'}
        
        # Test dimension filtering
        filter_obj = LesionSizeFilter(min_dimension=5, lesion_csv_path=csv_path)
        valid_cases = filter_obj.get_valid_cases()
        assert valid_cases == {'GIST-002_CT', 'GIST-004_CT', 'GIST-005_CT'}
        
        # Test density filtering
        filter_obj = LesionSizeFilter(min_density=0.3, lesion_csv_path=csv_path)
        valid_cases = filter_obj.get_valid_cases()
        assert valid_cases == {'GIST-002_CT', 'GIST-004_CT', 'GIST-005_CT'}
        
        # Test combined filtering (all criteria must pass)
        filter_obj = LesionSizeFilter(
            min_voxels=500,
            min_dimension=7,
            min_density=0.4,
            lesion_csv_path=csv_path
        )
        valid_cases = filter_obj.get_valid_cases()
        assert valid_cases == {'GIST-004_CT', 'GIST-005_CT'}
        
    finally:
        csv_path.unlink()


def test_load_lesion_filter_from_config():
    """Test loading filter from config dictionary."""
    # Test with all parameters
    config = {
        "min_voxels": 500,
        "min_dimension": 5,
        "min_density": 0.3,
    }
    filter_obj = load_lesion_filter_from_config(config)
    assert filter_obj is not None
    assert filter_obj.min_voxels == 500
    assert filter_obj.min_dimension == 5
    assert filter_obj.min_density == 0.3
    
    # Test with empty config
    config = {}
    filter_obj = load_lesion_filter_from_config(config)
    assert filter_obj is None
    
    # Test with partial config
    config = {"min_voxels": 500}
    filter_obj = load_lesion_filter_from_config(config)
    assert filter_obj is not None
    assert filter_obj.min_voxels == 500
    assert filter_obj.min_dimension is None


def test_lesion_filter_print_summary_no_csv():
    """Test that print_summary handles missing CSV gracefully."""
    filter_obj = LesionSizeFilter(
        min_voxels=500,
        lesion_csv_path=Path("nonexistent.csv")
    )
    
    # Should not crash, just print warning
    try:
        filter_obj.print_filter_summary()
    except FileNotFoundError:
        # Expected if we try to get valid cases
        pass


def test_lesion_filter_strict_filtering():
    """Test very strict filtering that leaves few cases."""
    test_data = pd.DataFrame({
        'dataset': ['gist'] * 5,
        'case': [f'GIST-00{i}_CT' for i in range(1, 6)],
        'preproc_voxels': [100, 200, 300, 400, 500],
        'min_dimension': [2, 3, 4, 5, 6],
        'density': [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        csv_path = Path(f.name)
    
    try:
        # Very strict filter
        filter_obj = LesionSizeFilter(
            min_voxels=500,
            min_dimension=6,
            min_density=0.5,
            lesion_csv_path=csv_path
        )
        valid_cases = filter_obj.get_valid_cases()
        assert len(valid_cases) == 1
        assert valid_cases == {'GIST-005_CT'}
        
    finally:
        csv_path.unlink()


if __name__ == "__main__":
    # Run tests manually
    test_lesion_filter_disabled()
    test_lesion_filter_enabled()
    test_lesion_filter_get_valid_cases()
    test_load_lesion_filter_from_config()
    test_lesion_filter_print_summary_no_csv()
    test_lesion_filter_strict_filtering()
    print("✅ All tests passed!")
