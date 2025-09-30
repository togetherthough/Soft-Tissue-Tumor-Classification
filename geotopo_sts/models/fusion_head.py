"""
Fusion modules and hierarchical classification heads
Includes gated fusion, Euclidean head, and hyperbolic (Poincaré) head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class GatedFusion(nn.Module):
    """
    Gated fusion module for combining multiple feature vectors.
    
    Uses gating mechanism: z = g ⊙ u + (1-g) ⊙ h
    where g is learned gate, u is transformed features, h is input
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256
    ):
        super().__init__()
        
        self.ln = nn.LayerNorm(in_channels)
        
        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
        
        # Transform network
        self.transform_net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Optional projection for residual
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of (B, F_i) tensors to fuse
        
        Returns:
            z_fused: (B, out_channels)
        """
        # Concatenate all features
        h = torch.cat(features, dim=-1)  # (B, sum(F_i))
        h = self.ln(h)
        
        # Gated fusion
        g = self.gate_net(h)  # (B, out_channels)
        u = self.transform_net(h)  # (B, out_channels)
        h_proj = self.residual_proj(h)
        
        z = g * u + (1 - g) * h_proj
        
        return z


class EuclideanHead(nn.Module):
    """
    Standard linear classification head in Euclidean space.
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 50
    ):
        super().__init__()
        self.classifier = nn.Linear(in_channels, num_classes)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, in_channels) fused features
        
        Returns:
            logits: (B, num_classes)
        """
        return self.classifier(z)


class HyperbolicHead(nn.Module):
    """
    Hyperbolic classification head using Poincaré ball model.
    
    Maps features to hyperbolic space for better hierarchical modeling.
    Requires geoopt library: pip install geoopt
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 50,
        c: float = 1.0
    ):
        super().__init__()
        
        try:
            import geoopt
            self.has_geoopt = True
        except ImportError:
            print("Warning: geoopt not installed. Using Euclidean fallback.")
            self.has_geoopt = False
            self.fallback = EuclideanHead(in_channels, num_classes)
            return
        
        self.c = c  # curvature parameter
        
        # Project to tangent space
        self.to_tangent = nn.Linear(in_channels, in_channels)
        
        # Hyperbolic linear layer (in gyrovector space)
        # For simplicity, project to Euclidean, then to hyperbolic
        self.classifier = nn.Linear(in_channels, num_classes)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, in_channels) Euclidean features
        
        Returns:
            logits: (B, num_classes) in hyperbolic space (projected to Euclidean for loss)
        """
        if not self.has_geoopt:
            return self.fallback(z)
        
        # Project to tangent space at origin
        z_tangent = self.to_tangent(z)
        
        # Exponential map to Poincaré ball
        z_hyp = self.exp_map(z_tangent)
        
        # Linear classifier (simplified, should use gyrovector operations)
        logits = self.classifier(z_hyp)
        
        return logits
    
    def exp_map(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from tangent space at origin to Poincaré ball.
        
        Args:
            v: (B, D) tangent vectors
        
        Returns:
            x: (B, D) points in Poincaré ball
        """
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
        sqrt_c = torch.sqrt(torch.tensor(self.c))
        
        x = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
        
        return x


class HierHead(nn.Module):
    """
    Wrapper that can use either Euclidean or Hyperbolic head.
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 50,
        head_type: str = 'euclidean'
    ):
        super().__init__()
        
        if head_type == 'euclidean':
            self.head = EuclideanHead(in_channels, num_classes)
        elif head_type == 'hyperbolic':
            self.head = HyperbolicHead(in_channels, num_classes)
        else:
            raise ValueError(f"Unknown head type: {head_type}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


def hierarchical_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ancestors: List[List[int]],
    lambda_h: float = 0.3
) -> torch.Tensor:
    """
    Hierarchical cross-entropy loss with ancestor smoothing.
    
    For each true class, also assign partial weight to ancestor classes.
    
    Args:
        logits: (B, K) predicted logits
        targets: (B,) true class indices
        ancestors: List of length K, where ancestors[i] is list of ancestor indices for class i
        lambda_h: weight for ancestor smoothing
    
    Returns:
        loss: scalar
    """
    B, K = logits.shape
    device = logits.device
    
    # Create soft targets
    soft_targets = torch.zeros_like(logits)
    
    for i, target in enumerate(targets):
        target_idx = target.item()
        anc = ancestors[target_idx] if target_idx < len(ancestors) else []
        
        if len(anc) == 0:
            # No ancestors, use hard target
            soft_targets[i, target_idx] = 1.0
        else:
            # Distribute weight between target and ancestors
            soft_targets[i, target_idx] = 1.0 - lambda_h
            
            ancestor_weight = lambda_h / len(anc)
            for a in anc:
                if a < K:
                    soft_targets[i, a] += ancestor_weight
    
    # Cross-entropy with soft targets
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    
    return loss


def compute_class_weights(
    class_counts: torch.Tensor,
    method: str = 'sqrt_inv'
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        class_counts: (K,) number of samples per class
        method: 'inv' (1/freq), 'sqrt_inv' (1/sqrt(freq)), or 'effective'
    
    Returns:
        weights: (K,) normalized class weights
    """
    if method == 'inv':
        weights = 1.0 / (class_counts + 1e-6)
    elif method == 'sqrt_inv':
        weights = 1.0 / (torch.sqrt(class_counts.float()) + 1e-6)
    elif method == 'effective':
        # Effective number of samples: (1 - beta^n) / (1 - beta)
        beta = 0.999
        effective_num = 1.0 - torch.pow(beta, class_counts.float())
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize
    weights = weights / weights.sum() * len(weights)
    
    return weights
