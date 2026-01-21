import torch
import torch.nn as nn
import numpy as np
from med3pipe.sam.core import average_pool_embedding

def test_pooling_match():
    print("Starting verification...", flush=True)
    
    # Create dummy embedding: Batch=1, Channels=4, Depth=4, Height=4, Width=4
    # Using random values to be sure
    torch.manual_seed(42)
    C, D, H, W = 4, 8, 8, 8
    embedding = torch.randn(1, C, D, H, W)
    
    # 1. Compute using Torch's AdaptiveAvgPool3d (Reference)
    # This is what the classification head uses
    gap_layer = nn.AdaptiveAvgPool3d(1)
    reference_tensor = gap_layer(embedding)  # (1, C, 1, 1, 1)
    # Flatten just like classification_head.py: x.view(x.size(0), -1)
    reference_flat = reference_tensor.view(embedding.size(0), -1) # (1, C)
    reference_numpy = reference_flat.squeeze(0).numpy() # (C,)
    
    print(f"Reference (AdaptiveAvgPool3d) shape: {reference_numpy.shape}")
    print(f"Reference values: {reference_numpy}")

    # 2. Compute using our modified average_pool_embedding
    # We pass a mask to ensure it is IGNORED
    dummy_mask = torch.ones(1, D, H, W) 
    # In ROI pooling, a mask of all ones would equal GAP, but let's try a partial mask
    # to prove it's being ignored.
    partial_mask = torch.zeros(1, D, H, W)
    partial_mask[:, :D//2, :, :] = 1.0
    
    result_numpy = average_pool_embedding(embedding, partial_mask)
    
    print(f"Function (average_pool_embedding) shape: {result_numpy.shape}")
    print(f"Function values: {result_numpy}")

    # Compare
    if np.allclose(reference_numpy, result_numpy, atol=1e-6):
        print("\n✅ MATCH CONFIRMED")
        print("The 'average_pool_embedding' function produces the exact same output")
        print("as 'nn.AdaptiveAvgPool3d(1)' used in the classification head.")
    else:
        print("\n❌ MISMATCH")
        print("The outputs differ.")

if __name__ == "__main__":
    test_pooling_match()
