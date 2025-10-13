"""Stage-1 Model: Coarse prediction + saliency generation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .swin3d import build_swin3d_tiny, build_swin3d_small


class Stage1Model(nn.Module):
    """Stage-1: Full volume → binary prediction + saliency map
    
    Architecture:
    - 3D Swin Transformer backbone (Tiny/Small)
    - Binary classification head (benign vs. malignant)
    - Saliency head for foreground detection
    - Modality embedding (CT vs MRI)
    
    For binary classification: n_classes = 2 (benign=0, malignant=1)
    """
    
    def __init__(
        self,
        n_classes: int = 2,  # Binary: benign vs. malignant
        backbone: str = 'swin3d_t',
        embed_dim: int = 96,
        saliency_channels: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            n_classes: Number of classes (2 for binary: benign vs. malignant)
            backbone: 'swin3d_t' or 'swin3d_s'
            embed_dim: Embedding dimension
            saliency_channels: Number of saliency channels (typically 1)
            dropout: Dropout rate
        """
        super().__init__()
        
        # Build backbone
        if backbone == 'swin3d_t':
            self.backbone = build_swin3d_tiny(in_chans=1, embed_dim=embed_dim)
        elif backbone == 'swin3d_s':
            self.backbone = build_swin3d_small(in_chans=1, embed_dim=embed_dim)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Get output dimensions
        self.num_features = self.backbone.num_features
        
        # Modality embedding
        self.modality_embed = nn.Embedding(2, self.num_features)  # 0=CT, 1=MRI
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, n_classes)
        )
        
        # Saliency head (operates on feature map)
        # We need to upsample from feature map to original resolution
        self.saliency_head = nn.Sequential(
            nn.Conv3d(self.num_features, 256, kernel_size=3, padding=1),
            nn.InstanceNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, saliency_channels, kernel_size=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        modality_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input volume (B, 1, D, H, W)
            modality_id: Modality indices (B,) - 0 for CT, 1 for MRI
            
        Returns:
            logits: Classification logits (B, n_classes)
            saliency: Saliency map (B, 1, D, H, W) in [0, 1]
        """
        # Extract features
        features, pooled = self.backbone(x)  # features: (B, C, D', H', W'), pooled: (B, C)
        
        # Add modality information to pooled features
        modality_emb = self.modality_embed(modality_id)  # (B, C)
        pooled = pooled + modality_emb
        
        # Classification
        logits = self.cls_head(pooled)  # (B, n_classes)
        
        # Saliency
        saliency = self.saliency_head(features)  # (B, 1, D', H', W')
        
        # Upsample saliency to input resolution
        saliency = F.interpolate(
            saliency,
            size=x.shape[-3:],
            mode='trilinear',
            align_corners=False
        )
        
        # Sigmoid to get [0, 1] range
        saliency = torch.sigmoid(saliency)
        
        return logits, saliency
    
    @torch.no_grad()
    def forward_saliency(
        self,
        x: torch.Tensor,
        modality_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for saliency generation only (no gradients)"""
        return self.forward(x, modality_id)
    
    def get_num_params(self) -> int:
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_stage1_model(
    n_classes: int,
    backbone: str = 'swin3d_t',
    embed_dim: int = 96,
    **kwargs
) -> Stage1Model:
    """Factory function for Stage-1 model
    
    Args:
        n_classes: Number of classes
        backbone: Backbone architecture
        embed_dim: Embedding dimension
        **kwargs: Additional arguments
        
    Returns:
        Stage1Model instance
    """
    return Stage1Model(
        n_classes=n_classes,
        backbone=backbone,
        embed_dim=embed_dim,
        **kwargs
    )
