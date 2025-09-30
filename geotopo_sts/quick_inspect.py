"""
Quick inspection of GIST data structure
Run this first to see what your data looks like!
"""

import sys
sys.path.insert(0, r'C:\Users\cahel\Desktop\Med3Tab-PFN')

from geotopo_sts.inspect_gist_structure import inspect_directory_structure

# Path to your GIST data
gist_root = r'C:\Users\cahel\Desktop\Med3Tab-PFN\data\gist'

print("\n" + "="*70)
print("🔍 GIST DATA STRUCTURE INSPECTOR")
print("="*70)

# Inspect the structure
inspect_directory_structure(gist_root, max_cases=5)

print("\n" + "="*70)
print("💡 TIP: Copy the file names shown above and update gist_data_loader.py")
print("="*70)
print("\nIf you see your actual file names, update lines 54-66 in gist_data_loader.py")
print("to include those exact file names.")
print("\n")
