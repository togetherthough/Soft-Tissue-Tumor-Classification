"""
Quick test to verify mesh features are being computed correctly.
"""

import sys
sys.path.insert(0, r'C:\Users\cahel\Desktop\Med3Tab-PFN')

import numpy as np
import yaml
from geotopo_sts.gist_data_loader import discover_gist_cases, load_gist_case
from geotopo_sts.dataio.preprocess import preprocess_case
from geotopo_sts.geometry import extract_mesh_from_mask, compute_mesh_node_features

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=" * 60)
print("MESH FEATURE VERIFICATION TEST")
print("=" * 60)

# Find a GIST case
gist_root = r'C:\Users\cahel\Desktop\Med3Tab-PFN\data\gist'
cases = discover_gist_cases(gist_root)
print(f"\nFound {len(cases)} cases")

# Load first case
print(f"\nLoading case: {cases[0]['case_id']}")
volume, mask, spacing = load_gist_case(cases[0])
print(f"  Volume shape: {volume.shape}")
print(f"  Mask shape: {mask.shape}")
print(f"  Spacing: {spacing}")

# Preprocess
print("\nPreprocessing...")
preprocessed = preprocess_case(volume, mask, spacing, 'ct', config['preprocessing'])
print(f"  Preprocessed volume: {preprocessed['volume'].shape}")
print(f"  Preprocessed mask: {preprocessed['mask'].shape}")

# Extract mesh
print("\nExtracting mesh...")
vertices, faces = extract_mesh_from_mask(
    preprocessed['mask'],
    spacing=preprocessed['spacing'],
    target_vertices=config['geometry']['mesh']['target_vertices']
)
print(f"  Vertices: {len(vertices)}")
print(f"  Faces: {len(faces)}")

# Compute node features
print("\nComputing mesh node features...")
node_features = compute_mesh_node_features(
    vertices, faces,
    preprocessed['volume'],
    preprocessed['mask'],
    preprocessed['rim'],
    spacing=preprocessed['spacing']
)

print(f"\n✅ NODE FEATURES SHAPE: {node_features.shape}")
print(f"   Expected: (N_vertices, 9)")
print(f"   Config expects: {config['geometry']['mesh']['feature_channels']} channels")

if node_features.shape[1] == config['geometry']['mesh']['feature_channels']:
    print("\n✅ ✅ ✅ MATCH! Data is correct!")
else:
    print(f"\n❌ MISMATCH!")
    print(f"   Data produces: {node_features.shape[1]} features")
    print(f"   Config expects: {config['geometry']['mesh']['feature_channels']} features")

# Show feature statistics
print("\nFeature statistics:")
feature_names = ['mean_curv', 'gauss_curv', 'shape_idx', 'curvedness', 
                 'surf_intensity', 'in_intensity', 'out_intensity', 
                 'dist_centroid', 'dist_rim']
for i, name in enumerate(feature_names):
    vals = node_features[:, i]
    print(f"  {i+1}. {name:20s}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
          f"min={vals.min():.3f}, max={vals.max():.3f}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
