"""Pooling modules for aggregating multiple crops into a single study embedding

Implements:
- Set Transformer (ISAB-based)
- Attention MIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiheadAttentionBlock(nn.Module):
    """Multihead Attention Block"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        """
        Args:
            query, key, value: (B, N, D)
        """
        attn_out, attn_weights = self.attention(query, key, value)
        out = self.norm(query + self.dropout(attn_out))
        return out, attn_weights


class InducedSetAttentionBlock(nn.Module):
    """Induced Set Attention Block (ISAB) from Set Transformer"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_inds: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_inds = num_inds
        
        # Learnable inducing points
        self.inducing_points = nn.Parameter(torch.randn(1, num_inds, dim))
        
        # Two attention blocks
        self.mab1 = MultiheadAttentionBlock(dim, num_heads, dropout)
        self.mab2 = MultiheadAttentionBlock(dim, num_heads, dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        B = x.shape[0]
        
        # Expand inducing points for batch
        I = self.inducing_points.expand(B, -1, -1)  # (B, num_inds, D)
        
        # MAB(I, X)
        H, _ = self.mab1(I, x, x)  # (B, num_inds, D)
        
        # MAB(X, H)
        out, _ = self.mab2(x, H, H)  # (B, N, D)
        
        # FFN
        out = out + self.ffn(self.norm(out))
        
        return out


class PoolingByMultiheadAttention(nn.Module):
    """Pooling by Multihead Attention (PMA) from Set Transformer"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_seeds: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_seeds = num_seeds
        
        # Learnable seed vectors
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, dim))
        
        self.mab = MultiheadAttentionBlock(dim, num_heads, dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, num_seeds, D)
        """
        B = x.shape[0]
        S = self.seeds.expand(B, -1, -1)
        out, attn_weights = self.mab(S, x, x)
        return out, attn_weights


class SetTransformer(nn.Module):
    """Set Transformer for permutation-invariant pooling
    
    Uses ISAB blocks followed by PMA for pooling.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_inds: int = 4,
        num_layers: int = 2,
        num_seeds: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # ISAB layers
        self.isab_layers = nn.ModuleList([
            InducedSetAttentionBlock(dim, num_heads, num_inds, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooling
        self.pma = PoolingByMultiheadAttention(dim, num_heads, num_seeds, dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, K, D) where K is number of crops
        Returns:
            (B, D) pooled representation
        """
        # ISAB layers
        for isab in self.isab_layers:
            x = isab(x)
        
        # Pool
        pooled, attn_weights = self.pma(x)  # (B, 1, D)
        pooled = pooled.squeeze(1)  # (B, D)
        
        return pooled


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning pooling
    
    Computes attention weights for each instance (crop) and pools via weighted sum.
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, K, D) where K is number of crops
        Returns:
            (B, D) pooled representation
        """
        # Compute attention weights
        attn_scores = self.attention(x)  # (B, K, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, K, 1)
        
        # Weighted sum
        pooled = torch.sum(attn_weights * x, dim=1)  # (B, D)
        
        return pooled


class GatedAttentionMIL(nn.Module):
    """Gated Attention MIL with more sophisticated attention mechanism"""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim
        
        # Attention V
        self.attention_V = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh()
        )
        
        # Attention U
        self.attention_U = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Attention weights
        self.attention_w = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, K, D)
        Returns:
            (B, D)
        """
        # Gated attention
        A_V = self.attention_V(x)  # (B, K, hidden_dim)
        A_U = self.attention_U(x)  # (B, K, hidden_dim)
        A = A_V * A_U  # Element-wise gating
        A = self.dropout(A)
        
        attn_scores = self.attention_w(A)  # (B, K, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted sum
        pooled = torch.sum(attn_weights * x, dim=1)  # (B, D)
        
        return pooled


def build_pooling(pooling_type: str, dim: int, **kwargs):
    """Factory function for pooling modules
    
    Args:
        pooling_type: 'set_transformer', 'attention_mil', or 'gated_attention_mil'
        dim: Embedding dimension
        **kwargs: Additional arguments
    
    Returns:
        Pooling module
    """
    if pooling_type == 'set_transformer':
        return SetTransformer(
            dim=dim,
            num_heads=kwargs.get('num_heads', 4),
            num_inds=kwargs.get('num_inds', 4),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif pooling_type == 'attention_mil':
        return AttentionMIL(
            dim=dim,
            hidden_dim=kwargs.get('hidden_dim', dim),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif pooling_type == 'gated_attention_mil':
        return GatedAttentionMIL(
            dim=dim,
            hidden_dim=kwargs.get('hidden_dim', dim),
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
