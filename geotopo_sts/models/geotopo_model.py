"""
End-to-end GeoTopo-STS model
Integrates all pathways: voxel, geometry, topology
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from .voxels import Vox3DResNet, Vox3DMamba
from .mesh_gnn import MeshEGNN, MeshE3NN
from .topo_mlp import TopoMLP
from .fusion_head import GatedFusion, HierHead


class GeoTopoSTS(nn.Module):
    """
    Complete GeoTopo-STS model for soft tissue sarcoma classification.
    
    Combines:
    - Voxel pathway: 3D CNN/Mamba with ROI pooling
    - Geometry pathway: Mesh GNN
    - Topology pathway: PH features MLP
    - Fusion & hierarchical classification
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.use_rim = config.get('use_rim', True)
        self.use_mesh = config.get('use_mesh', True)
        self.use_topology = config.get('use_topology', True)
        self.use_skeleton = config.get('use_skeleton', False)
        
        # Voxel pathway
        voxel_cfg = config.get('voxel', {})
        voxel_backbone = voxel_cfg.get('backbone', 'resnet')
        
        if voxel_backbone == 'resnet':
            self.voxel_encoder = Vox3DResNet(
                in_channels=voxel_cfg.get('in_channels', 1),
                width=voxel_cfg.get('width', 32),
                out_channels=voxel_cfg.get('out_channels', 256)
            )
        elif voxel_backbone == 'mamba':
            self.voxel_encoder = Vox3DMamba(
                in_channels=voxel_cfg.get('in_channels', 1),
                width=voxel_cfg.get('width', 32),
                out_channels=voxel_cfg.get('out_channels', 256)
            )
        else:
            raise ValueError(f"Unknown voxel backbone: {voxel_backbone}")
        
        voxel_out_dim = voxel_cfg.get('out_channels', 256)
        
        # Mesh pathway
        fusion_dims = []
        if self.use_mesh:
            mesh_cfg = config.get('mesh_gnn', {})
            gnn_type = mesh_cfg.get('type', 'egnn')
            
            if gnn_type == 'egnn':
                self.mesh_encoder = MeshEGNN(
                    in_channels=mesh_cfg.get('in_channels', 8),
                    hidden_channels=mesh_cfg.get('hidden', 128),
                    num_layers=mesh_cfg.get('layers', 4),
                    out_channels=mesh_cfg.get('out_channels', 256)
                )
            elif gnn_type == 'e3nn':
                self.mesh_encoder = MeshE3NN(
                    in_channels=mesh_cfg.get('in_channels', 8),
                    hidden_channels=mesh_cfg.get('hidden', 128),
                    num_layers=mesh_cfg.get('layers', 3),
                    out_channels=mesh_cfg.get('out_channels', 256)
                )
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            mesh_out_dim = mesh_cfg.get('out_channels', 256)
            fusion_dims.append(mesh_out_dim)
        
        # Topology pathway
        if self.use_topology:
            topo_cfg = config.get('topology_mlp', {})
            self.topo_encoder = TopoMLP(
                in_channels=topo_cfg.get('in_channels', 128),
                hidden_channels=topo_cfg.get('hidden', 128),
                out_channels=topo_cfg.get('out_channels', 64),
                dropout=topo_cfg.get('dropout', 0.2)
            )
            topo_out_dim = topo_cfg.get('out_channels', 64)
            fusion_dims.append(topo_out_dim)
        
        # Compute total fusion input dimension
        # Voxel: 3 features (tumor, rim, global) OR 2 if not using rim
        voxel_features = 3 if self.use_rim else 2
        total_fusion_dim = voxel_features * voxel_out_dim + sum(fusion_dims)
        
        # Fusion module
        fusion_cfg = config.get('fusion', {})
        fusion_out_dim = fusion_cfg.get('out_channels', 256)
        
        self.fusion = GatedFusion(
            in_channels=total_fusion_dim,
            out_channels=fusion_out_dim
        )
        
        # Classification head
        head_cfg = config.get('head', {})
        self.head = HierHead(
            in_channels=fusion_out_dim,
            num_classes=head_cfg.get('n_classes', 50),
            head_type=head_cfg.get('type', 'euclidean')
        )
    
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass through all pathways.
        
        Args:
            batch: dict containing:
                - 'volume': (B, C, D, H, W)
                - 'mask': (B, 1, D, H, W)
                - 'rim': (B, 1, D, H, W)
                - 'mesh_data': dict with node_features, pos, edge_index (optional)
                - 'ph_features': (B, F) (optional)
        
        Returns:
            logits: (B, num_classes)
        """
        features_to_fuse = []
        
        # 1. Voxel pathway
        z_tumor, z_rim, z_global = self.voxel_encoder(
            batch['volume'],
            batch['mask'],
            batch['rim']
        )
        
        features_to_fuse.extend([z_tumor, z_global])
        if self.use_rim:
            features_to_fuse.append(z_rim)
        
        # 2. Mesh pathway
        if self.use_mesh and batch.get('mesh_data') is not None:
            mesh_data = batch['mesh_data']
            
            # Handle batched graphs
            if isinstance(mesh_data, dict):
                # Single graph or pre-batched
                z_mesh = self.mesh_encoder(
                    mesh_data['features'],
                    mesh_data['vertices'],
                    mesh_data['edge_index'],
                    batch=mesh_data.get('batch')
                )
                
                # If single graph per batch item, ensure shape is (B, D)
                if z_mesh.size(0) == 1 and z_tumor.size(0) > 1:
                    z_mesh = z_mesh.expand(z_tumor.size(0), -1)
            else:
                # List of graphs
                z_mesh_list = []
                for md in mesh_data:
                    z = self.mesh_encoder(
                        md['features'],
                        md['vertices'],
                        md['edge_index']
                    )
                    z_mesh_list.append(z.squeeze(0))
                z_mesh = torch.stack(z_mesh_list, dim=0)
            
            features_to_fuse.append(z_mesh)
        
        # 3. Topology pathway
        if self.use_topology and batch.get('ph_features') is not None:
            z_topo = self.topo_encoder(batch['ph_features'])
            features_to_fuse.append(z_topo)
        
        # 4. Fusion
        z_fused = self.fusion(features_to_fuse)
        
        # 5. Classification
        logits = self.head(z_fused)
        
        return logits
    
    def get_embeddings(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from each pathway for analysis.
        
        Returns:
            dict with embeddings from each pathway
        """
        embeddings = {}
        
        # Voxel
        z_tumor, z_rim, z_global = self.voxel_encoder(
            batch['volume'], batch['mask'], batch['rim']
        )
        embeddings['voxel_tumor'] = z_tumor
        embeddings['voxel_rim'] = z_rim
        embeddings['voxel_global'] = z_global
        
        # Mesh
        if self.use_mesh and batch.get('mesh_data') is not None:
            mesh_data = batch['mesh_data']
            z_mesh = self.mesh_encoder(
                mesh_data['features'],
                mesh_data['vertices'],
                mesh_data['edge_index'],
                batch=mesh_data.get('batch')
            )
            embeddings['mesh'] = z_mesh
        
        # Topology
        if self.use_topology and batch.get('ph_features') is not None:
            z_topo = self.topo_encoder(batch['ph_features'])
            embeddings['topology'] = z_topo
        
        # Fused
        features_to_fuse = [z_tumor, z_global]
        if self.use_rim:
            features_to_fuse.append(z_rim)
        if 'mesh' in embeddings:
            features_to_fuse.append(embeddings['mesh'])
        if 'topology' in embeddings:
            features_to_fuse.append(embeddings['topology'])
        
        z_fused = self.fusion(features_to_fuse)
        embeddings['fused'] = z_fused
        
        return embeddings
