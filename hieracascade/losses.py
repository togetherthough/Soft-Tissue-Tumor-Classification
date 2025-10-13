"""Loss functions for HieraCascade-STS"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights (C,)
            gamma: Focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (B, C)
            targets: Class labels (B,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HierarchicalConsistencyLoss(nn.Module):
    """KL divergence loss for hierarchical consistency
    
    Ensures that fine-grained predictions are consistent with coarse predictions.
    """
    
    def __init__(
        self,
        fine_to_coarse_map: Dict[int, int],
        n_fine: int,
        n_coarse: int
    ):
        """
        Args:
            fine_to_coarse_map: Mapping from fine class idx to coarse family idx
            n_fine: Number of fine classes
            n_coarse: Number of coarse classes
        """
        super().__init__()
        
        self.n_fine = n_fine
        self.n_coarse = n_coarse
        
        # Create mapping matrix: (n_coarse, n_fine)
        # mapping[c, f] = 1 if fine class f belongs to coarse class c
        self.register_buffer(
            'mapping',
            torch.zeros(n_coarse, n_fine)
        )
        
        for fine_idx, coarse_idx in fine_to_coarse_map.items():
            self.mapping[coarse_idx, fine_idx] = 1.0
        
        # Normalize to get probabilities
        self.mapping = self.mapping / (self.mapping.sum(dim=1, keepdim=True) + 1e-8)
    
    def forward(
        self,
        logits_fine: torch.Tensor,
        logits_coarse: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits_fine: Fine-grained logits (B, n_fine)
            logits_coarse: Coarse logits (B, n_coarse)
            
        Returns:
            KL divergence loss
        """
        # Get probabilities
        p_fine = F.softmax(logits_fine, dim=1)  # (B, n_fine)
        p_coarse = F.softmax(logits_coarse, dim=1)  # (B, n_coarse)
        
        # Aggregate fine predictions to coarse level
        # q_coarse[b, c] = sum over fine classes in family c of p_fine[b, f]
        q_coarse = torch.matmul(p_fine, self.mapping.T)  # (B, n_coarse)
        
        # KL(q || p) where q is from fine, p is from coarse head
        # KL(q || p) = sum_c q_c * log(q_c / p_c)
        kl_loss = F.kl_div(
            p_coarse.log(),
            q_coarse,
            reduction='batchmean',
            log_target=False
        )
        
        return kl_loss


class Stage1Loss(nn.Module):
    """Combined loss for Stage-1 training"""
    
    def __init__(
        self,
        n_classes: int,
        sparsity_weight: float = 1e-3,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 0.0
    ):
        """
        Args:
            n_classes: Number of classes
            sparsity_weight: Weight for saliency sparsity regularization
            class_weights: Optional class weights for CE loss
            focal_gamma: If > 0, use Focal Loss with this gamma
        """
        super().__init__()
        self.sparsity_weight = sparsity_weight
        
        if focal_gamma > 0:
            self.cls_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(
        self,
        logits: torch.Tensor,
        saliency: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: Classification logits (B, n_classes)
            saliency: Saliency map (B, 1, D, H, W)
            targets: Ground truth labels (B,)
            
        Returns:
            Dictionary with 'total', 'ce', 'sparsity' losses
        """
        # Classification loss
        loss_ce = self.cls_loss(logits, targets)
        
        # Sparsity regularization (L1 on saliency)
        loss_sparsity = saliency.mean()
        
        # Total loss
        loss_total = loss_ce + self.sparsity_weight * loss_sparsity
        
        return {
            'total': loss_total,
            'ce': loss_ce,
            'sparsity': loss_sparsity
        }


class Stage2Loss(nn.Module):
    """Combined hierarchical loss for Stage-2 training"""
    
    def __init__(
        self,
        n_fine: int,
        n_coarse: int,
        fine_to_coarse_map: Dict[int, int],
        alpha: float = 0.3,
        beta: float = 0.1,
        class_weights_fine: Optional[torch.Tensor] = None,
        class_weights_coarse: Optional[torch.Tensor] = None,
        focal_gamma: float = 0.0
    ):
        """
        Args:
            n_fine: Number of fine classes
            n_coarse: Number of coarse classes
            fine_to_coarse_map: Mapping from fine to coarse indices
            alpha: Weight for coarse CE loss
            beta: Weight for consistency loss
            class_weights_fine: Optional class weights for fine CE
            class_weights_coarse: Optional class weights for coarse CE
            focal_gamma: If > 0, use Focal Loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        # Fine-grained loss
        if focal_gamma > 0:
            self.loss_fine = FocalLoss(alpha=class_weights_fine, gamma=focal_gamma)
        else:
            self.loss_fine = nn.CrossEntropyLoss(weight=class_weights_fine)
        
        # Coarse loss
        self.loss_coarse = nn.CrossEntropyLoss(weight=class_weights_coarse)
        
        # Consistency loss
        self.loss_consistency = HierarchicalConsistencyLoss(
            fine_to_coarse_map, n_fine, n_coarse
        )
    
    def forward(
        self,
        logits_fine: torch.Tensor,
        logits_coarse: torch.Tensor,
        targets_fine: torch.Tensor,
        targets_coarse: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits_fine: Fine logits (B, n_fine)
            logits_coarse: Coarse logits (B, n_coarse)
            targets_fine: Fine labels (B,)
            targets_coarse: Coarse labels (B,)
            
        Returns:
            Dictionary with loss components
        """
        # Fine-grained CE
        L_f = self.loss_fine(logits_fine, targets_fine)
        
        # Coarse CE
        L_c = self.loss_coarse(logits_coarse, targets_coarse)
        
        # Hierarchical consistency
        L_cons = self.loss_consistency(logits_fine, logits_coarse)
        
        # Total loss
        L_total = L_f + self.alpha * L_c + self.beta * L_cons
        
        return {
            'total': L_total,
            'fine': L_f,
            'coarse': L_c,
            'consistency': L_cons
        }


def build_stage1_loss(
    n_classes: int,
    sparsity_weight: float = 1e-3,
    class_weights: Optional[List[float]] = None,
    focal_gamma: float = 0.0
) -> Stage1Loss:
    """Factory for Stage-1 loss"""
    
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    return Stage1Loss(
        n_classes=n_classes,
        sparsity_weight=sparsity_weight,
        class_weights=class_weights,
        focal_gamma=focal_gamma
    )


def build_stage2_loss(
    n_fine: int,
    n_coarse: int,
    fine_to_coarse_map: Dict[int, int],
    alpha: float = 0.3,
    beta: float = 0.1,
    class_weights_fine: Optional[List[float]] = None,
    class_weights_coarse: Optional[List[float]] = None,
    focal_gamma: float = 0.0
) -> Stage2Loss:
    """Factory for Stage-2 loss"""
    
    if class_weights_fine is not None:
        class_weights_fine = torch.tensor(class_weights_fine, dtype=torch.float32)
    
    if class_weights_coarse is not None:
        class_weights_coarse = torch.tensor(class_weights_coarse, dtype=torch.float32)
    
    return Stage2Loss(
        n_fine=n_fine,
        n_coarse=n_coarse,
        fine_to_coarse_map=fine_to_coarse_map,
        alpha=alpha,
        beta=beta,
        class_weights_fine=class_weights_fine,
        class_weights_coarse=class_weights_coarse,
        focal_gamma=focal_gamma
    )
