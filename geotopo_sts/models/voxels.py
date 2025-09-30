"""
3D Voxel pathway models
Includes 3D ResNet and Mamba-based architectures with ROI-aware pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock3D(nn.Module):
    """3D Convolutional block with InstanceNorm and GELU"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        
        # Residual connection
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + identity
        out = self.act(out)
        
        return out


class Vox3DResNet(nn.Module):
    """
    3D ResNet encoder with ROI-aware pooling.
    
    Outputs three feature vectors:
    - z_tumor: tumor ROI pooled features
    - z_rim: peritumoral rim pooled features
    - z_global: global average pooled features
    """
    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        out_channels: int = 256,
        depth: str = '18'  # '18' or '34'
    ):
        super().__init__()
        
        # Stem: downsample by 2
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, width, 7, stride=2, padding=3, bias=False),
            nn.InstanceNorm3d(width),
            nn.GELU()
        )
        
        # Layers
        self.layer1 = ConvBlock3D(width, width)
        self.layer2 = nn.Sequential(
            nn.Conv3d(width, 2*width, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            ConvBlock3D(2*width, 2*width)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(2*width, 4*width, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            ConvBlock3D(4*width, 4*width)
        )
        
        # Optional layer4 for deeper network
        if depth == '34':
            self.layer4 = nn.Sequential(
                nn.Conv3d(4*width, 8*width, 3, stride=2, padding=1, bias=False),
                nn.GELU(),
                ConvBlock3D(8*width, 8*width)
            )
            final_width = 8*width
        else:
            self.layer4 = nn.Identity()
            final_width = 4*width
        
        # Projection to output dimension
        self.proj = nn.Linear(final_width, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rim: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) input volume
            mask: (B, 1, D, H, W) tumor mask
            rim: (B, 1, D, H, W) peritumoral rim mask
        
        Returns:
            z_tumor: (B, out_channels)
            z_rim: (B, out_channels)
            z_global: (B, out_channels)
        """
        # Forward pass
        f = self.stem(x)           # (B, W, D/2, H/2, W/2)
        f = self.layer1(f)         # (B, W, D/2, H/2, W/2)
        f = self.layer2(f)         # (B, 2W, D/4, H/4, W/4)
        f = self.layer3(f)         # (B, 4W, D/8, H/8, W/8)
        f = self.layer4(f)         # (B, final_W, D/?, H/?, W/?)
        
        B, C, d, h, w = f.shape
        
        # ROI-aware pooling
        def pool_with_mask(feat, weight_mask):
            """Weighted average pooling"""
            # Resize mask to feature map size
            wmask = F.interpolate(
                weight_mask.float(),
                size=(d, h, w),
                mode='trilinear',
                align_corners=False
            )
            wmask = wmask.clamp(min=0.0)
            
            # Weighted sum
            weighted_sum = (feat * wmask).sum(dim=(2, 3, 4))  # (B, C)
            weight_total = wmask.sum(dim=(2, 3, 4)) + 1e-6   # (B, 1)
            
            pooled = weighted_sum / weight_total
            return self.proj(pooled)  # (B, out_channels)
        
        z_tumor = pool_with_mask(f, mask)
        z_rim = pool_with_mask(f, rim)
        z_global = self.proj(f.mean(dim=(2, 3, 4)))  # Global average pooling
        
        return z_tumor, z_rim, z_global


class Vox3DMamba(nn.Module):
    """
    3D Mamba-based encoder (state-space model for efficient long-range).
    
    This is a placeholder. For full implementation, use mamba-ssm library:
    - Stem: 3D CNN to tokenize
    - Mamba blocks on flattened spatial sequence
    - ROI-aware pooling
    
    For now, falls back to CNN+Transformer-like architecture.
    """
    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        out_channels: int = 256,
        num_blocks: int = 4
    ):
        super().__init__()
        
        # CNN stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, stride=2, padding=1),
            nn.InstanceNorm3d(width),
            nn.GELU(),
            nn.Conv3d(width, 2*width, 3, stride=2, padding=1),
            nn.InstanceNorm3d(2*width),
            nn.GELU(),
            nn.Conv3d(2*width, 4*width, 3, stride=2, padding=1),
            nn.InstanceNorm3d(4*width),
            nn.GELU()
        )
        
        # Placeholder: use self-attention (in production, use Mamba)
        self.mamba_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=4*width,
                nhead=8,
                dim_feedforward=8*width,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_blocks)
        ])
        
        self.proj = nn.Linear(4*width, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rim: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W)
            mask, rim: (B, 1, D, H, W)
        
        Returns:
            z_tumor, z_rim, z_global: each (B, out_channels)
        """
        # Stem
        f = self.stem(x)  # (B, 4W, D/8, H/8, W/8)
        B, C, d, h, w = f.shape
        
        # Flatten spatial dimensions for sequence modeling
        f_seq = f.view(B, C, -1).permute(0, 2, 1)  # (B, L, C) where L=d*h*w
        
        # Apply Mamba/Transformer blocks
        for block in self.mamba_blocks:
            f_seq = block(f_seq)
        
        # Reshape back to spatial
        f = f_seq.permute(0, 2, 1).view(B, C, d, h, w)
        
        # ROI-aware pooling (same as ResNet)
        def pool_with_mask(feat, weight_mask):
            wmask = F.interpolate(weight_mask.float(), size=(d, h, w), mode='trilinear', align_corners=False)
            wmask = wmask.clamp(min=0.0)
            weighted_sum = (feat * wmask).sum(dim=(2, 3, 4))
            weight_total = wmask.sum(dim=(2, 3, 4)) + 1e-6
            return self.proj(weighted_sum / weight_total)
        
        z_tumor = pool_with_mask(f, mask)
        z_rim = pool_with_mask(f, rim)
        z_global = self.proj(f.mean(dim=(2, 3, 4)))
        
        return z_tumor, z_rim, z_global
