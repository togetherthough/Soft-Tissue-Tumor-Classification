"""
Helper script to inspect GIST dataset structure
Use this to understand how your data is organized
"""

from pathlib import Path
import sys


def inspect_directory_structure(root_dir: str, max_cases: int = 3):
    """
    Inspect and print the structure of GIST dataset.
    
    Args:
        root_dir: Root directory of GIST data
        max_cases: Number of cases to show in detail
    """
    root = Path(root_dir)
    
    if not root.exists():
        print(f"❌ Directory not found: {root}")
        return
    
    print(f"\n📁 Inspecting: {root.absolute()}")
    print("=" * 60)
    
    # Find all potential case directories
    case_patterns = ["GIST-*_CT", "gist-*_CT", "GIST-*", "gist-*"]
    all_cases = []
    
    for pattern in case_patterns:
        cases = list(root.glob(pattern))
        if cases:
            all_cases.extend(cases)
            print(f"\n✓ Found {len(cases)} folders matching '{pattern}'")
    
    if not all_cases:
        print("\n❌ No GIST case folders found!")
        print("\nContents of root directory:")
        for item in sorted(root.iterdir())[:20]:
            print(f"  {'📁' if item.is_dir() else '📄'} {item.name}")
        return
    
    # Remove duplicates and sort
    all_cases = sorted(set(all_cases))
    
    print(f"\n📊 Total unique case folders: {len(all_cases)}")
    print("\n" + "=" * 60)
    print("DETAILED STRUCTURE (first {} cases):".format(max_cases))
    print("=" * 60)
    
    # Inspect first few cases in detail
    for i, case_dir in enumerate(all_cases[:max_cases], 1):
        print(f"\n{i}. {case_dir.name}/")
        
        # List all contents
        contents = sorted(case_dir.iterdir())
        for item in contents:
            if item.is_dir():
                print(f"   📁 {item.name}/")
                # Show contents of subdirectory
                sub_contents = sorted(item.iterdir())
                for sub_item in sub_contents[:10]:  # Limit to 10 items
                    print(f"      {'📁' if sub_item.is_dir() else '📄'} {sub_item.name}")
                if len(sub_contents) > 10:
                    print(f"      ... and {len(sub_contents) - 10} more")
            else:
                # Show file with size
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"   📄 {item.name} ({size_mb:.2f} MB)")
    
    # Summary of file types found
    print("\n" + "=" * 60)
    print("FILE TYPE SUMMARY (across all cases):")
    print("=" * 60)
    
    file_types = {}
    for case_dir in all_cases:
        for file_path in case_dir.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix
                file_types[ext] = file_types.get(ext, 0) + 1
    
    for ext, count in sorted(file_types.items()):
        print(f"  {ext if ext else '(no extension)'}: {count} files")
    
    # Check for common NIFTI patterns
    print("\n" + "=" * 60)
    print("NIFTI FILE PATTERNS:")
    print("=" * 60)
    
    nifti_patterns = {
        "image.nii.gz": 0,
        "image.nii": 0,
        "mask.nii.gz": 0,
        "mask.nii": 0,
        "segmentation.nii.gz": 0,
        "segmentation.nii": 0,
        "ct.nii.gz": 0,
        "ct.nii": 0,
    }
    
    for case_dir in all_cases:
        for pattern, count in nifti_patterns.items():
            matches = list(case_dir.rglob(pattern))
            if matches:
                nifti_patterns[pattern] += len(matches)
    
    for pattern, count in nifti_patterns.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {pattern}: {count} found")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    if any(count > 0 for count in nifti_patterns.values()):
        found_patterns = [p for p, c in nifti_patterns.items() if c > 0]
        print(f"\n✓ Found NIfTI files! Patterns detected:")
        for p in found_patterns:
            print(f"  - {p}")
        print("\nThe data loader should work with these files.")
    else:
        print("\n❌ No standard NIfTI file names found!")
        print("\nPlease check:")
        print("  1. Are your files in NIfTI format (.nii or .nii.gz)?")
        print("  2. What are the exact file names?")
        print("  3. Run this script to see the actual structure above.")
    
    print("\n" + "=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect GIST dataset structure')
    parser.add_argument('--root', type=str, default='../data/gist',
                       help='Root directory of GIST data')
    parser.add_argument('--max-cases', type=int, default=3,
                       help='Number of cases to show in detail')
    args = parser.parse_args()
    
    inspect_directory_structure(args.root, args.max_cases)


if __name__ == '__main__':
    main()
