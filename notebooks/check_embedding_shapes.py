import torch
from pathlib import Path

# Check train embeddings
train_dir = Path(r"C:\Users\cahel\Desktop\Med3Tab-PFN\SAM-Med3D-main\SAM-Med3D-main\features\lipo\ct_LIPO_train")
val_dir = Path(r"C:\Users\cahel\Desktop\Med3Tab-PFN\SAM-Med3D-main\SAM-Med3D-main\features\lipo\ct_LIPO")

print("=" * 60)
print("TRAIN EMBEDDINGS")
print("=" * 60)
train_shapes = set()
for pt in sorted(train_dir.glob("*_embedding.pt")):
    d = torch.load(str(pt), map_location="cpu")
    emb = d["embedding"]
    train_shapes.add(emb.shape)
    if len(train_shapes) > 1:
        print(f"  {pt.name}: {emb.shape}")

print(f"Unique train shapes: {train_shapes}")

print("\n" + "=" * 60)
print("VALIDATION EMBEDDINGS")
print("=" * 60)
val_shapes = set()
for pt in sorted(val_dir.glob("*_embedding.pt")):
    d = torch.load(str(pt), map_location="cpu")
    emb = d["embedding"]
    val_shapes.add(emb.shape)
    if len(val_shapes) > 1:
        print(f"  {pt.name}: {emb.shape}")

print(f"Unique val shapes: {val_shapes}")

print("\n" + "=" * 60)
if len(train_shapes) > 1 or len(val_shapes) > 1:
    print("❌ SHAPE MISMATCH DETECTED!")
    print(f"  Train shapes: {train_shapes}")
    print(f"  Val shapes: {val_shapes}")
else:
    print("✅ All shapes match!")
    print(f"  Train shape: {train_shapes}")
    print(f"  Val shape: {val_shapes}")
