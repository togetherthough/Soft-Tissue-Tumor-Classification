"""
Evaluation script for GeoTopo-STS
Includes robustness testing, ablation studies, and hierarchical metrics
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from .models import GeoTopoSTS
from .dataio import GeoTopoDataset, get_val_transforms
from .train import compute_metrics, compute_ece


class Evaluator:
    """Evaluation manager for GeoTopo-STS"""
    
    def __init__(
        self,
        model: GeoTopoSTS,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
        output_dir: str = './eval_results'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_classes = config['head']['n_classes']
        self.class_names = config.get('class_names', [f'Class_{i}' for i in range(self.num_classes)])
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Run full evaluation on test set.
        
        Returns comprehensive metrics including:
        - Overall accuracy, balanced accuracy, macro F1
        - Per-class metrics
        - Confusion matrix
        - Calibration (ECE, Brier)
        """
        all_preds = []
        all_targets = []
        all_probs = []
        all_case_ids = []
        
        for batch in tqdm(self.test_loader, desc='Evaluating'):
            batch = self._to_device(batch)
            
            logits = self.model(batch)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch['label'].cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            if 'case_id' in batch:
                all_case_ids.extend(batch['case_id'])
        
        # Concatenate
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        
        # Compute metrics
        metrics = compute_metrics(all_preds, all_targets, self.num_classes)
        
        # Calibration
        metrics['ece'] = compute_ece(all_probs, all_targets)
        metrics['brier'] = self._compute_brier(all_probs, all_targets)
        
        # Per-class metrics
        metrics['per_class'] = self._compute_per_class_metrics(all_preds, all_targets)
        
        # Save results
        self._save_results(metrics, all_preds, all_targets, all_probs, all_case_ids)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(metrics['confusion_matrix'])
        
        return metrics
    
    def _compute_brier(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """Compute Brier score"""
        # Convert targets to one-hot
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(targets)), targets] = 1
        
        # Brier score
        brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
        return brier
    
    def _compute_per_class_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict[str, List[float]]:
        """Compute per-class precision, recall, F1"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, labels=range(self.num_classes), zero_division=0
        )
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        }
    
    def evaluate_robustness(self) -> Dict[str, Any]:
        """
        Test model robustness to:
        - Rotation perturbations
        - Missing modalities
        - Intensity variations
        """
        results = {}
        
        # 1. Rotation robustness
        print("Testing rotation robustness...")
        rotation_angles = self.config.get('evaluation', {}).get('robustness_tests', {}).get('rotation_angles', [0, 45, 90])
        results['rotation'] = self._test_rotation_robustness(rotation_angles)
        
        # 2. Missing modality (if applicable)
        if self.config.get('evaluation', {}).get('robustness_tests', {}).get('missing_modality', False):
            print("Testing missing modality robustness...")
            results['missing_modality'] = self._test_missing_modality()
        
        # Save
        with open(self.output_dir / 'robustness_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    @torch.no_grad()
    def _test_rotation_robustness(self, angles: List[float]) -> Dict[str, float]:
        """Test with different rotation angles"""
        from scipy.ndimage import rotate
        
        results = {}
        
        for angle in angles:
            all_preds = []
            all_targets = []
            
            for batch in tqdm(self.test_loader, desc=f'Rotation {angle}°'):
                batch = self._to_device(batch)
                
                # Rotate volume
                if angle != 0:
                    volume = batch['volume'].cpu().numpy()
                    mask = batch['mask'].cpu().numpy()
                    rim = batch['rim'].cpu().numpy()
                    
                    # Rotate around Z-axis
                    volume_rot = rotate(volume, angle, axes=(3, 4), reshape=False, order=1)
                    mask_rot = rotate(mask, angle, axes=(3, 4), reshape=False, order=0)
                    rim_rot = rotate(rim, angle, axes=(3, 4), reshape=False, order=0)
                    
                    batch['volume'] = torch.from_numpy(volume_rot).to(self.device)
                    batch['mask'] = torch.from_numpy(mask_rot).to(self.device)
                    batch['rim'] = torch.from_numpy(rim_rot).to(self.device)
                
                logits = self.model(batch)
                preds = logits.argmax(dim=-1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch['label'].cpu().numpy())
            
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            
            metrics = compute_metrics(all_preds, all_targets, self.num_classes)
            results[f'angle_{angle}'] = {
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'balanced_accuracy': metrics['balanced_accuracy']
            }
        
        return results
    
    @torch.no_grad()
    def _test_missing_modality(self) -> Dict[str, float]:
        """Test with missing mesh/topology features"""
        # Test with mesh only, topology only, both
        conditions = {
            'full': {'use_mesh': True, 'use_topology': True},
            'no_mesh': {'use_mesh': False, 'use_topology': True},
            'no_topology': {'use_mesh': True, 'use_topology': False},
            'voxel_only': {'use_mesh': False, 'use_topology': False}
        }
        
        results = {}
        
        for name, config in conditions.items():
            # Temporarily modify model config
            original_use_mesh = self.model.use_mesh
            original_use_topo = self.model.use_topology
            
            self.model.use_mesh = config['use_mesh']
            self.model.use_topology = config['use_topology']
            
            all_preds = []
            all_targets = []
            
            for batch in tqdm(self.test_loader, desc=f'Condition: {name}'):
                batch = self._to_device(batch)
                
                # Zero out features if missing
                if not config['use_mesh']:
                    batch['mesh_data'] = None
                if not config['use_topology']:
                    batch['ph_features'] = None
                
                logits = self.model(batch)
                preds = logits.argmax(dim=-1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch['label'].cpu().numpy())
            
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            
            metrics = compute_metrics(all_preds, all_targets, self.num_classes)
            results[name] = {
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1']
            }
            
            # Restore
            self.model.use_mesh = original_use_mesh
            self.model.use_topology = original_use_topo
        
        return results
    
    def run_ablations(self) -> Dict[str, Any]:
        """
        Run ablation studies to assess component importance.
        
        Tests:
        - Tumor-only vs +rim
        - +PH features
        - +Mesh geometry
        - Hierarchical loss effect
        """
        ablation_configs = {
            'tumor_only': {'use_rim': False, 'use_mesh': False, 'use_topology': False},
            'tumor+rim': {'use_rim': True, 'use_mesh': False, 'use_topology': False},
            'tumor+rim+ph': {'use_rim': True, 'use_mesh': False, 'use_topology': True},
            'tumor+rim+mesh': {'use_rim': True, 'use_mesh': True, 'use_topology': False},
            'full': {'use_rim': True, 'use_mesh': True, 'use_topology': True}
        }
        
        results = {}
        
        for name, config in ablation_configs.items():
            print(f"\nAblation: {name}")
            
            # Modify model config
            self.model.use_rim = config['use_rim']
            self.model.use_mesh = config['use_mesh']
            self.model.use_topology = config['use_topology']
            
            # Evaluate
            metrics = self.evaluate()
            
            results[name] = {
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'balanced_accuracy': metrics['balanced_accuracy']
            }
        
        # Save ablation results
        with open(self.output_dir / 'ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot ablation comparison
        self._plot_ablation_comparison(results)
        
        return results
    
    def extract_embeddings(self, save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from all pathways for visualization/analysis.
        
        Returns:
            dict with embeddings arrays
        """
        all_embeddings = {
            'voxel_tumor': [],
            'voxel_rim': [],
            'voxel_global': [],
            'mesh': [],
            'topology': [],
            'fused': [],
            'targets': [],
            'case_ids': []
        }
        
        for batch in tqdm(self.test_loader, desc='Extracting embeddings'):
            batch = self._to_device(batch)
            
            embeddings = self.model.get_embeddings(batch)
            
            for key in ['voxel_tumor', 'voxel_rim', 'voxel_global']:
                all_embeddings[key].append(embeddings[key].cpu().numpy())
            
            if 'mesh' in embeddings:
                all_embeddings['mesh'].append(embeddings['mesh'].cpu().numpy())
            if 'topology' in embeddings:
                all_embeddings['topology'].append(embeddings['topology'].cpu().numpy())
            
            all_embeddings['fused'].append(embeddings['fused'].cpu().numpy())
            all_embeddings['targets'].append(batch['label'].cpu().numpy())
            
            if 'case_id' in batch:
                all_embeddings['case_ids'].extend(batch['case_id'])
        
        # Concatenate
        for key in all_embeddings:
            if key != 'case_ids' and len(all_embeddings[key]) > 0:
                all_embeddings[key] = np.concatenate(all_embeddings[key])
        
        # Save
        if save_path is None:
            save_path = self.output_dir / 'embeddings.npz'
        
        np.savez(save_path, **all_embeddings)
        print(f"Saved embeddings to {save_path}")
        
        return all_embeddings
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
    
    def _plot_ablation_comparison(self, results: Dict[str, Dict[str, float]]):
        """Plot ablation study results"""
        conditions = list(results.keys())
        metrics = ['accuracy', 'macro_f1', 'balanced_accuracy']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            values = [results[cond][metric] for cond in conditions]
            
            axes[idx].bar(range(len(conditions)), values, color='steelblue')
            axes[idx].set_xticks(range(len(conditions)))
            axes[idx].set_xticklabels(conditions, rotation=45, ha='right')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_comparison.png', dpi=300)
        plt.close()
    
    def _save_results(
        self,
        metrics: Dict[str, Any],
        preds: np.ndarray,
        targets: np.ndarray,
        probs: np.ndarray,
        case_ids: List[str]
    ):
        """Save evaluation results to files"""
        # Metrics JSON
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif key != 'confusion_matrix':
                metrics_serializable[key] = value
        
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        # Predictions CSV
        import pandas as pd
        
        df = pd.DataFrame({
            'case_id': case_ids if case_ids else range(len(preds)),
            'true_label': targets,
            'predicted_label': preds,
            'confidence': probs.max(axis=1)
        })
        
        df.to_csv(self.output_dir / 'predictions.csv', index=False)
        
        # Full probabilities
        np.save(self.output_dir / 'probabilities.npy', probs)
    
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        for key in ['volume', 'mask', 'rim', 'label', 'ph_features']:
            if key in batch:
                batch[key] = batch[key].to(self.device)
        
        if 'mesh_data' in batch and batch['mesh_data'] is not None:
            if isinstance(batch['mesh_data'], dict):
                for key in ['vertices', 'features', 'edge_index']:
                    if key in batch['mesh_data']:
                        batch['mesh_data'][key] = batch['mesh_data'][key].to(self.device)
        
        return batch


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate GeoTopo-STS')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--ablations', action='store_true', help='Run ablation studies')
    parser.add_argument('--robustness', action='store_true', help='Run robustness tests')
    parser.add_argument('--embeddings', action='store_true', help='Extract embeddings')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = GeoTopoSTS(config['model'])
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test dataset
    test_dataset = GeoTopoDataset(
        args.data,
        split='test',
        transform=get_val_transforms(),
        config=config
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config['model'],
        device=args.device,
        output_dir=args.output
    )
    
    # Run evaluation
    print("Running standard evaluation...")
    metrics = evaluator.evaluate()
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"Brier: {metrics['brier']:.4f}")
    print(f"{'='*60}\n")
    
    # Optional: ablations
    if args.ablations:
        print("\nRunning ablation studies...")
        evaluator.run_ablations()
    
    # Optional: robustness
    if args.robustness:
        print("\nRunning robustness tests...")
        evaluator.evaluate_robustness()
    
    # Optional: embeddings
    if args.embeddings:
        print("\nExtracting embeddings...")
        evaluator.extract_embeddings()
    
    print(f"\nAll results saved to {args.output}")


if __name__ == '__main__':
    main()
