"""3D Swin Transformer backbone for medical imaging

Simplified implementation based on Swin Transformer architecture adapted for 3D volumes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding"""
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_chans: int = 1,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Projection
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, D', H', W', embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.permute(0, 2, 3, 4, 1)  # (B, D', H', W', embed_dim)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """MLP with GELU activation"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention3D(nn.Module):
    """Window-based Multi-head Self Attention for 3D"""
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        """
        Args:
            x: (num_windows*B, window_size*window_size*window_size, C)
        """
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block for 3D"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (7, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x, D, H, W):
        """
        Args:
            x: (B, D*H*W, C)
            D, H, W: spatial dimensions
        """
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)
        
        # Cyclic shift (simplified - no actual shifting for base impl)
        shifted_x = x
        
        # Partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, D, H, W)
        x = shifted_x
        
        x = x.view(B, D * H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def window_partition(self, x, window_size):
        """Partition into non-overlapping windows"""
        B, D, H, W, C = x.shape
        x = x.view(
            B,
            D // window_size[0], window_size[0],
            H // window_size[1], window_size[1],
            W // window_size[2], window_size[2],
            C
        )
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = windows.view(-1, window_size[0], window_size[1], window_size[2], C)
        return windows
    
    def window_reverse(self, windows, window_size, D, H, W):
        """Reverse window partitioning"""
        B = int(windows.shape[0] / (D * H * W / np.prod(window_size)))
        x = windows.view(
            B,
            D // window_size[0], H // window_size[1], W // window_size[2],
            window_size[0], window_size[1], window_size[2], -1
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, D, H, W, -1)
        return x


class PatchMerging3D(nn.Module):
    """Patch Merging Layer for downsampling"""
    
    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)
    
    def forward(self, x, D, H, W):
        """
        Args:
            x: (B, D*H*W, C)
        """
        B, L, C = x.shape
        assert L == D * H * W
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0
        
        x = x.view(B, D, H, W, C)
        
        # Downsample by 2x in each dimension (8 patches merged)
        x0 = x[:, 0::2, 0::2, 0::2, :]  # (B, D/2, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # (B, D/2, H/2, W/2, 8*C)
        x = x.view(B, -1, 8 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class BasicLayer3D(nn.Module):
    """A basic Swin Transformer layer for one stage"""
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        downsample: Optional[nn.Module] = None,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else tuple(w // 2 for w in window_size),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        self.downsample = downsample
    
    def forward(self, x, D, H, W):
        for blk in self.blocks:
            x = blk(x, D, H, W)
        
        if self.downsample is not None:
            x = self.downsample(x, D, H, W)
            D, H, W = D // 2, H // 2, W // 2
        
        return x, D, H, W


class SwinTransformer3D(nn.Module):
    """3D Swin Transformer backbone"""
    
    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_size: Tuple[int, int, int] = (4, 4, 4)
    ):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer3D(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,
                norm_layer=norm_layer
            )
            self.layers.append(layer)
        
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            features: (B, C', D', H', W') feature map
            pooled: (B, num_features) global features
        """
        x = self.patch_embed(x)  # (B, D', H', W', C)
        B, D, H, W, C = x.shape
        x = x.view(B, D * H * W, C)
        
        # Forward through layers
        for layer in self.layers:
            x, D, H, W = layer(x, D, H, W)
        
        # Get feature map and pooled features
        x = self.norm(x)  # (B, D*H*W, C)
        
        # Global pooling
        pooled = self.avgpool(x.transpose(1, 2))  # (B, C, 1)
        pooled = torch.flatten(pooled, 1)  # (B, C)
        
        # Reshape for feature map
        features = x.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        
        return features, pooled


def build_swin3d_tiny(**kwargs):
    """Build Swin-T 3D model"""
    model = SwinTransformer3D(
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7),
        **kwargs
    )
    return model


def build_swin3d_small(**kwargs):
    """Build Swin-S 3D model"""
    model = SwinTransformer3D(
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7),
        **kwargs
    )
    return model


def build_swin3d_base(**kwargs):
    """Build Swin-B 3D model"""
    model = SwinTransformer3D(
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=(7, 7, 7),
        **kwargs
    )
    return model
