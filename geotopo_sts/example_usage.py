"""
Example usage of GeoTopo-STS
Demonstrates basic workflow from data loading to evaluation
"""

import torch
import yaml
from pathlib import Path

# Example 1: Load and inspect a preprocessed sample
def example_load_data():
    """Load and visualize preprocessed data"""
    from geotopo_sts.dataio import GeoTopoDataset
    
    # Create dataset
    dataset = GeoTopoDataset(
        data_root='./preprocessed_data',
        split='train',
        load_mesh=True,
        load_topology=True
    )
    
    # Load one sample
    sample = dataset[0]
    
    print("Sample contents:")
    print(f"  Volume shape: {sample['volume'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Rim shape: {sample['rim'].shape}")
    print(f"  Label: {sample['label']}")
    
    if sample['mesh_data'] is not None:
        print(f"  Mesh vertices: {sample['mesh_data']['vertices'].shape}")
        print(f"  Mesh features: {sample['mesh_data']['features'].shape}")
        print(f"  Mesh edges: {sample['mesh_data']['edge_index'].shape}")
    
    if sample['ph_features'] is not None:
        print(f"  PH features: {sample['ph_features'].shape}")
    
    return sample


# Example 2: Create and test model
def example_model_forward():
    """Create model and run forward pass"""
    from geotopo_sts.models import GeoTopoSTS
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = GeoTopoSTS(config['model'])
    model.eval()
    
    # Create dummy batch
    batch = {
        'volume': torch.randn(2, 1, 160, 160, 160),
        'mask': torch.rand(2, 1, 160, 160, 160) > 0.5,
        'rim': torch.rand(2, 1, 160, 160, 160) > 0.7,
        'mesh_data': {
            'vertices': torch.randn(8000, 3),
            'features': torch.randn(8000, 9),
            'edge_index': torch.randint(0, 8000, (2, 20000))
        },
        'ph_features': torch.randn(2, 128),
        'label': torch.tensor([0, 1])
    }
    
    # Forward pass
    with torch.no_grad():
        logits = model(batch)
    
    print(f"\nModel output shape: {logits.shape}")
    print(f"Predicted classes: {logits.argmax(dim=-1)}")
    
    # Get embeddings
    embeddings = model.get_embeddings(batch)
    print("\nEmbeddings:")
    for key, value in embeddings.items():
        print(f"  {key}: {value.shape}")
    
    return model, logits


# Example 3: Quick training loop
def example_train_one_epoch():
    """Train for one epoch (demonstration)"""
    from torch.utils.data import DataLoader
    from geotopo_sts.dataio import GeoTopoDataset, get_train_transforms
    from geotopo_sts.models import GeoTopoSTS
    import torch.nn.functional as F
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Dataset and loader
    dataset = GeoTopoDataset(
        data_root='./preprocessed_data',
        split='train',
        transform=get_train_transforms(config.get('augmentation'))
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # Model
    model = GeoTopoSTS(config['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Train one epoch
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 5:  # Just 5 batches for demo
            break
        
        # Forward
        logits = model(batch)
        loss = F.cross_entropy(logits, batch['label'])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    
    print(f"\nAverage loss: {total_loss / 5:.4f}")


# Example 4: Evaluate pretrained model
def example_evaluate():
    """Load pretrained model and evaluate"""
    from geotopo_sts.models import GeoTopoSTS
    from geotopo_sts.eval import Evaluator
    from geotopo_sts.dataio import GeoTopoDataset
    from torch.utils.data import DataLoader
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = GeoTopoSTS(config['model'])
    checkpoint = torch.load('./outputs/best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test dataset
    test_dataset = GeoTopoDataset(
        data_root='./preprocessed_data',
        split='test'
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config['model'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    print("="*60)
    
    return metrics


# Example 5: Preprocess a single case
def example_preprocess_case():
    """Preprocess one case from raw data"""
    import numpy as np
    from geotopo_sts.dataio.preprocess import preprocess_case
    from geotopo_sts.geometry import extract_mesh_from_mask, compute_mesh_node_features
    from geotopo_sts.topology import extract_ph_features
    
    # Load raw data (example with numpy arrays)
    volume = np.random.randn(200, 200, 200)  # Replace with actual load
    mask = np.random.rand(200, 200, 200) > 0.8
    spacing = (1.5, 1.5, 1.5)
    
    # Preprocess
    preprocessed = preprocess_case(
        volume, mask, spacing,
        modality='mri',
        config={
            'target_spacing': [1.5, 1.5, 1.5],
            'rim_radius_mm': 10.0,
            'crop_padding': 25,
            'crop_size': [160, 160, 160],
            'bias_correction': False,  # Set True for real MRI
            'percentile_clip': [1, 99]
        }
    )
    
    print("Preprocessed data:")
    print(f"  Volume: {preprocessed['volume'].shape}")
    print(f"  Mask: {preprocessed['mask'].shape}")
    print(f"  Rim: {preprocessed['rim'].shape}")
    
    # Extract mesh
    vertices, faces = extract_mesh_from_mask(
        preprocessed['mask'],
        spacing=preprocessed['spacing'],
        target_vertices=8000
    )
    
    print(f"\nMesh extraction:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    
    # Extract topology features
    ph_features = extract_ph_features(
        preprocessed['mask'],
        preprocessed['rim'],
        preprocessed['volume'],
        spacing=preprocessed['spacing']
    )
    
    print(f"\nTopology features: {ph_features.shape}")
    
    return preprocessed


# Example 6: Ablation study
def example_ablation():
    """Run quick ablation to test components"""
    from geotopo_sts.models import GeoTopoSTS
    from geotopo_sts.eval import Evaluator
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test different configurations
    configs_to_test = {
        'tumor_only': {'use_rim': False, 'use_mesh': False, 'use_topology': False},
        'tumor+rim': {'use_rim': True, 'use_mesh': False, 'use_topology': False},
        'full': {'use_rim': True, 'use_mesh': True, 'use_topology': True}
    }
    
    results = {}
    
    for name, ablation_config in configs_to_test.items():
        print(f"\nTesting: {name}")
        
        # Update config
        model_config = config['model'].copy()
        model_config.update(ablation_config)
        
        # Create model (in practice, load checkpoint)
        model = GeoTopoSTS(model_config)
        
        print(f"  use_rim: {model.use_rim}")
        print(f"  use_mesh: {model.use_mesh}")
        print(f"  use_topology: {model.use_topology}")
        
        # In practice: evaluate on test set
        # results[name] = evaluator.evaluate()
    
    return results


if __name__ == '__main__':
    print("GeoTopo-STS Example Usage")
    print("="*60)
    
    # Run examples (comment out as needed)
    
    print("\n1. Loading data...")
    # example_load_data()
    
    print("\n2. Model forward pass...")
    example_model_forward()
    
    print("\n3. Training one epoch...")
    # example_train_one_epoch()
    
    print("\n4. Evaluation...")
    # example_evaluate()
    
    print("\n5. Preprocessing...")
    # example_preprocess_case()
    
    print("\n6. Ablation study...")
    # example_ablation()
    
    print("\n" + "="*60)
    print("Examples complete! See function docstrings for details.")
    print("="*60)
