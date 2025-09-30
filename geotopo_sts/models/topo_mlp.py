"""
Topology pathway: MLP for persistent homology features
"""

import torch
import torch.nn as nn


class TopoMLP(nn.Module):
    """
    Simple MLP for processing persistent homology features.
    
    Input: PCA-reduced persistence images (typically 128-dim)
    Output: topology embedding
    """
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels) persistence homology features
        
        Returns:
            z: (B, out_channels) topology embedding
        """
        return self.net(x)
