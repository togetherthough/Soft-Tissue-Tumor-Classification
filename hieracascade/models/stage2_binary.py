"""Stage-2 Model: Fine-grained hierarchical classification from crops"""

import torch
import torch.nn as nn
from typing import Tuple

from .swin3d import build_swin3d_small, build_swin3d_base
from .pooling import build_pooling


class Stage2Model(nn.Module):
    """Stage-2: Crops → binary classification via MIL
    
    Architecture:
    - 3D Swin Transformer encoder for each crop
    - Set-based pooling (Set Transformer or Attention MIL)
    - Binary classification head (benign vs. malignant)
    - Modality embedding
    
    For binary classification: n_classes = 2 (benign=0, malignant=1)
    """
    
    def __init__(
        self,
        n_classes: int = 2,  # Binary: benign vs. malignant
        backbone: str = 'swin3d_b',
        embed_dim: int = 768,
        pooling: str = 'set_transformer',
        pooling_kwargs: dict = None,
        dropout: float = 0.1
    ):
        """
        Args:
            n_classes: Number of classes (2 for binary: benign vs. malignant)
            backbone: 'swin3d_s' or 'swin3d_b'
            embed_dim: Embedding dimension for projection
            pooling: 'set_transformer', 'attention_mil', or 'gated_attention_mil'
            pooling_kwargs: Additional arguments for pooling module
            dropout: Dropout rate
        """
        super().__init__()
        
        # Build backbone encoder
        if backbone == 'swin3d_s':
            self.encoder = build_swin3d_small(in_chans=1)
        elif backbone == 'swin3d_b':
            self.encoder = build_swin3d_base(in_chans=1)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Get encoder output dimension
        self.encoder_dim = self.encoder.num_features
        
        # Projection to common embedding space
        self.proj = nn.Sequential(
            nn.Linear(self.encoder_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Modality embedding
        self.modality_embed = nn.Embedding(2, embed_dim)  # 0=CT, 1=MRI
        
        # Pooling module
        pooling_kwargs = pooling_kwargs or {}
        self.pool = build_pooling(pooling, dim=embed_dim, **pooling_kwargs)
        
        # Binary classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes)
        )
    
    def encode_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """Encode individual crops
        
        Args:
            crops: (B*K, 1, S, S, S)
            
        Returns:
            (B*K, embed_dim) crop embeddings
        """
        _, pooled = self.encoder(crops)  # (B*K, encoder_dim)
        embeddings = self.proj(pooled)  # (B*K, embed_dim)
        return embeddings
    
    def forward(
        self,
        crops: torch.Tensor,
        modality_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            crops: Cropped volumes (B, K, 1, S, S, S)
            modality_id: Modality indices (B,) - 0 for CT, 1 for MRI
            
        Returns:
            logits: Binary logits (B, n_classes) - typically (B, 2) for benign vs. malignant
            study_embedding: Pooled study embedding (B, embed_dim)
        """
        B, K = crops.shape[:2]
        
        # Reshape crops for batch processing
        crops_flat = crops.view(B * K, 1, *crops.shape[-3:])  # (B*K, 1, S, S, S)
        
        # Encode each crop
        crop_embeddings = self.encode_crops(crops_flat)  # (B*K, embed_dim)
        
        # Reshape back to (B, K, embed_dim)
        crop_embeddings = crop_embeddings.view(B, K, -1)
        
        # Add modality information
        modality_emb = self.modality_embed(modality_id)  # (B, embed_dim)
        modality_emb = modality_emb.unsqueeze(1)  # (B, 1, embed_dim)
        crop_embeddings = crop_embeddings + modality_emb  # Broadcasting
        
        # Pool crops to study-level embedding (MIL aggregation)
        study_embedding = self.pool(crop_embeddings)  # (B, embed_dim)
        
        # Binary classification head
        logits = self.head(study_embedding)  # (B, n_classes)
        
        return logits, study_embedding
    
    def get_num_params(self) -> int:
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_stage2_model(
    n_classes: int = 2,  # Binary: benign vs. malignant
    backbone: str = 'swin3d_b',
    pooling: str = 'set_transformer',
    **kwargs
) -> Stage2Model:
    """Factory function for Stage-2 model
    
    Args:
        n_classes: Number of classes (2 for binary: benign vs. malignant)
        backbone: Backbone architecture
        pooling: Pooling strategy
        **kwargs: Additional arguments
        
    Returns:
        Stage2Model instance
    """
    return Stage2Model(
        n_classes=n_classes,
        backbone=backbone,
        pooling=pooling,
        **kwargs
    )
