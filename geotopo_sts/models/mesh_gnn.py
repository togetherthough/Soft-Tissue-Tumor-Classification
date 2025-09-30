"""
Mesh GNN models using equivariant graph networks
Supports EGNN (E(n)-equivariant) and E3NN (SE(3)-equivariant with spherical harmonics)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class EGNNLayer(nn.Module):
    """
    E(n)-Equivariant Graph Neural Network layer.
    
    Updates node features and coordinates in an equivariant manner.
    Based on Satorras et al. 2021 (https://arxiv.org/abs/2102.09844)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        edge_attr_dim: int = 0
    ):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Edge model: predicts edge features and coordinate updates
        edge_input_dim = 2 * in_channels + 1 + edge_attr_dim  # h_i, h_j, ||x_i - x_j||, edge_attr
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU()
        )
        
        # Coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1, bias=False)
        )
        
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (N, F) node features
            x: (N, 3) node coordinates
            edge_index: (2, E) edge indices
            edge_attr: (E, D) edge attributes (optional)
        
        Returns:
            h_new: (N, out_channels) updated node features
            x_new: (N, 3) updated coordinates
        """
        src, dst = edge_index
        
        # Edge features
        h_i, h_j = h[src], h[dst]
        x_i, x_j = x[src], x[dst]
        
        rel_pos = x_i - x_j  # (E, 3)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # (E, 1)
        
        # Edge input: [h_i, h_j, dist, edge_attr]
        edge_input = [h_i, h_j, dist]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        edge_feat = torch.cat(edge_input, dim=-1)
        
        # Edge MLP
        m_ij = self.edge_mlp(edge_feat)  # (E, hidden)
        
        # Coordinate update
        coord_weight = self.coord_mlp(m_ij)  # (E, 1)
        coord_diff = coord_weight * rel_pos / (dist + 1e-6)  # (E, 3)
        
        # Aggregate to destination nodes
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, dst, coord_diff)
        x_new = x + x_update
        
        # Node update
        # Aggregate messages
        m_i = torch.zeros(h.size(0), m_ij.size(1), device=h.device)
        m_i.index_add_(0, dst, m_ij)
        
        node_input = torch.cat([h, m_i], dim=-1)
        h_new = self.node_mlp(node_input)
        
        # Residual
        if self.out_channels == self.in_channels:
            h_new = h_new + h
        
        return h_new, x_new


class MeshEGNN(nn.Module):
    """
    Mesh encoder using stacked EGNN layers with attention pooling.
    """
    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 128,
        num_layers: int = 4,
        out_channels: int = 256
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])
        
        # Attention pooling
        self.attn_weights = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # Output projection
        self.out_proj = nn.Linear(hidden_channels, out_channels)
    
    def forward(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: (N, F) node features
            pos: (N, 3) positions in mm
            edge_index: (2, E) edge connectivity
            batch: (N,) batch assignment for multiple graphs
        
        Returns:
            graph_embedding: (num_graphs, out_channels) if batch provided, else (1, out_channels)
        """
        # Project input
        h = self.input_proj(node_features)
        x = pos
        
        # EGNN layers
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        
        # Attention pooling
        attn_logits = self.attn_weights(h).squeeze(-1)  # (N,)
        
        if batch is not None:
            # Pool per graph
            num_graphs = batch.max().item() + 1
            graph_embeddings = []
            
            for i in range(num_graphs):
                mask = (batch == i)
                h_i = h[mask]
                attn_i = torch.softmax(attn_logits[mask], dim=0)
                
                pooled = (attn_i.unsqueeze(-1) * h_i).sum(dim=0)
                graph_embeddings.append(pooled)
            
            z = torch.stack(graph_embeddings, dim=0)  # (num_graphs, hidden)
        else:
            # Single graph
            attn = torch.softmax(attn_logits, dim=0)
            z = (attn.unsqueeze(-1) * h).sum(dim=0, keepdim=True)  # (1, hidden)
        
        # Output projection
        z = self.out_proj(z)
        
        return z


class MeshE3NN(nn.Module):
    """
    SE(3)-equivariant mesh encoder using e3nn library.
    
    This is a placeholder. Full implementation requires e3nn:
    pip install e3nn
    
    Uses irreducible representations (irreps) and tensor products.
    """
    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 128,
        num_layers: int = 3,
        out_channels: int = 256
    ):
        super().__init__()
        
        try:
            from e3nn import o3
            from e3nn.nn import FullyConnectedNet
        except ImportError:
            print("Warning: e3nn not installed. MeshE3NN will use EGNN fallback.")
            # Fallback to EGNN
            self.fallback = MeshEGNN(in_channels, hidden_channels, num_layers, out_channels)
            self.use_e3nn = False
            return
        
        self.use_e3nn = True
        
        # Define irreps (irreducible representations)
        # Type-0: scalars (e.g., curvature, intensity)
        # Type-1: vectors (e.g., normals)
        
        self.in_irreps = o3.Irreps(f"{in_channels}x0e")  # All scalars for now
        self.hidden_irreps = o3.Irreps(f"{hidden_channels//2}x0e + {hidden_channels//2}x1o")
        self.out_irreps = o3.Irreps(f"{out_channels}x0e")
        
        # TODO: Implement e3nn message passing layers
        # For now, just use fallback
        self.fallback = MeshEGNN(in_channels, hidden_channels, num_layers, out_channels)
    
    def forward(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Same interface as MeshEGNN"""
        return self.fallback(node_features, pos, edge_index, batch)
