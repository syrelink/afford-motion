import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
# 假设 hilbert.py 在同级目录下
from .hilbert import encode
from typing import Optional


# ============================================================
# 基础组件
# ============================================================

class PointMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        # [修改点 1] 移除内部残差连接
        # 这里只计算 Mamba 的变换量 (delta)，残差在外部循环中加
        # 这样可以方便地将 forward 和 backward 的结果叠加到同一个 x 上
        return self.mamba(self.norm(x))


class MambaFeatureEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.GELU(),
            nn.Linear(128, out_channels),
            nn.LayerNorm(out_channels)
        )

    def forward(self, x):
        return self.net(x)


class ContactPointMamba(nn.Module):
    def __init__(self, arch_cfg, contact_dim, point_feat_dim, text_feat_dim, time_emb_dim):
        super().__init__()

        self.trans_dim = getattr(arch_cfg, 'trans_dim', 256)
        self.depth = getattr(arch_cfg, 'depth', 8)
        self.last_dim = getattr(arch_cfg, 'last_dim', 256)

        # 输入维度
        self.in_channels = contact_dim + point_feat_dim + text_feat_dim + time_emb_dim + 3

        # Embedding
        self.embedding = MambaFeatureEncoder(self.in_channels, self.trans_dim)

        # Mamba Backbone
        # 为了节省显存，这里正反向共用同一套权重 (Shared Weights Bi-Mamba)
        # 这是一个常见的 trick，既能有双向感知，又不增加参数量
        self.layers = nn.ModuleList([
            PointMambaBlock(self.trans_dim) for _ in range(self.depth)
        ])

        # Output Projection
        self.out_proj = nn.Linear(self.trans_dim, self.last_dim)

    def forward(self, x, point_feat, language_feat, time_embedding, **kwargs):
        bs, num_points, _ = x.shape
        xyz = kwargs['c_pc_xyz']  # [B, N, 3]

        # --- 1. 特征融合 ---
        text_rep = language_feat.repeat(1, num_points, 1)
        time_rep = time_embedding.repeat(1, num_points, 1)

        features_list = [x, text_rep, time_rep, xyz]
        if point_feat is not None:
            features_list.append(point_feat)

        fusion_feat = torch.cat(features_list, dim=-1)  # [B, N, C_in]
        x_emb = self.embedding(fusion_feat)  # [B, N, trans_dim]

        # --- 2. Hilbert 空间排序 ---
        # 2.1 归一化坐标到 [0, 1]
        xyz_min = xyz.min(dim=1, keepdim=True)[0]
        xyz_max = xyz.max(dim=1, keepdim=True)[0]
        xyz_norm = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8) * 0.999

        # 2.2 量化到整数空间
        num_bits = 16
        xyz_int = (xyz_norm * (2 ** num_bits - 1)).long()

        # 2.3 生成 Hilbert 编码
        xyz_int_flat = xyz_int.view(-1, 3)
        hilbert_code = encode(xyz_int_flat, num_dims=3, num_bits=num_bits)

        # 2.4 获取排序索引
        hilbert_code = hilbert_code.view(bs, num_points)
        sort_idx = torch.argsort(hilbert_code, dim=1)
        unsort_idx = torch.argsort(sort_idx, dim=1)

        # 2.5 Apply Gather (乱序 -> 有序)
        idx_expand = sort_idx.unsqueeze(-1).expand(-1, -1, self.trans_dim)
        x_sorted = torch.gather(x_emb, 1, idx_expand)

        # --- 3. Bi-Mamba Scanning (核心修复部分) ---
        for layer in self.layers:
            # [Step A] 正向扫描
            # 因为 PointMambaBlock 不再返回 x + out，这里得到的是纯增量
            out_fwd = layer(x_sorted)

            # [Step B] 反向扫描
            x_rev = torch.flip(x_sorted, dims=[1])
            out_bwd = layer(x_rev)
            out_bwd = torch.flip(out_bwd, dims=[1])  # 翻转回正向顺序

            # [Step C] 正确的残差连接 (Residual Connection)
            # 公式: x_new = x_old + fwd_delta + bwd_delta
            # 这样既融合了双向信息，又保证了信号幅度稳定
            x_sorted = x_sorted + out_fwd + out_bwd

        # --- 4. 还原顺序 (Scatter) ---
        unsort_idx_expand = unsort_idx.unsqueeze(-1).expand(-1, -1, self.trans_dim)
        x_out = torch.gather(x_sorted, 1, unsort_idx_expand)

        # --- 5. 输出 ---
        return self.out_proj(x_out)


# ============================================================
# Hybrid Mamba = Mamba (局部精度) + Perceiver Bottleneck (全局一致性)
# ============================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for language conditioning."""

    def __init__(self, in_dim: int, cond_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, in_dim)
        self.beta_proj = nn.Linear(cond_dim, in_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - input features
            cond: [B, C] - condition features (language + time)
        Returns:
            [B, N, D] - modulated features
        """
        gamma = self.gamma_proj(cond).unsqueeze(1)  # [B, 1, D]
        beta = self.beta_proj(cond).unsqueeze(1)    # [B, 1, D]
        return gamma * x + beta


class ConditionedMambaBlock(nn.Module):
    """Mamba block with FiLM-based language conditioning."""

    def __init__(self, dim: int, cond_dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.film = FiLMLayer(dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            cond: [B, C]
        """
        out = self.mamba(self.norm(x))
        out = self.film(out, cond)
        return out


class PerceiverBottleneck(nn.Module):
    """
    Perceiver-style latent bottleneck for global information aggregation.
    Uses cross-attention to compress point features into latent tokens,
    then broadcasts back to enhance global consistency.
    """

    def __init__(
        self,
        point_dim: int,
        latent_dim: int,
        num_latents: int = 128,
        num_heads: int = 8,
        num_self_attn_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Learnable latent array
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)

        # Point -> Latent (Encode): latents attend to points
        self.encode_cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.point_to_latent_proj = nn.Linear(point_dim, latent_dim)
        self.encode_norm_q = nn.LayerNorm(latent_dim)
        self.encode_norm_kv = nn.LayerNorm(latent_dim)

        # Process: self-attention on latent tokens
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=latent_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_self_attn_layers)
        ])

        # Latent -> Point (Decode): points attend to latents
        self.decode_cross_attn = nn.MultiheadAttention(
            embed_dim=point_dim,
            kdim=latent_dim,
            vdim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.decode_norm_q = nn.LayerNorm(point_dim)
        self.decode_norm_kv = nn.LayerNorm(latent_dim)

        # Output projection with residual gate
        self.gate = nn.Sequential(
            nn.Linear(point_dim * 2, point_dim),
            nn.Sigmoid()
        )

    def forward(self, point_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_feat: [B, N, D_point] - point-wise features from Mamba
        Returns:
            [B, N, D_point] - enhanced features with global consistency
        """
        bs = point_feat.shape[0]

        # Expand latent tokens for batch
        latents = self.latent_tokens.expand(bs, -1, -1)  # [B, L, D_latent]

        # === Encode: Latents attend to Points ===
        point_proj = self.point_to_latent_proj(point_feat)  # [B, N, D_latent]
        q = self.encode_norm_q(latents)
        kv = self.encode_norm_kv(point_proj)
        latents, _ = self.encode_cross_attn(q, kv, kv)  # [B, L, D_latent]

        # === Process: Self-attention on latents ===
        for layer in self.self_attn_layers:
            latents = layer(latents)

        # === Decode: Points attend to Latents ===
        q = self.decode_norm_q(point_feat)
        kv = self.decode_norm_kv(latents)
        global_feat, _ = self.decode_cross_attn(q, kv, kv)  # [B, N, D_point]

        # === Gated Residual Fusion ===
        gate_input = torch.cat([point_feat, global_feat], dim=-1)
        gate_weight = self.gate(gate_input)
        output = point_feat + gate_weight * global_feat

        return output


class MultiScanMamba(nn.Module):
    """
    Multi-scan Mamba encoder that uses multiple ordering strategies
    to capture different spatial relationships.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        depth: int = 4,
        num_scans: int = 2,  # hilbert + reverse
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.dim = dim
        self.num_scans = num_scans
        self.depth = depth

        # Shared Mamba layers (parameter efficient)
        self.layers = nn.ModuleList([
            ConditionedMambaBlock(dim, cond_dim, d_state, d_conv, expand)
            for _ in range(depth)
        ])

        # Scan fusion
        self.scan_fusion = nn.Sequential(
            nn.Linear(dim * num_scans, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

    def _hilbert_sort(self, xyz: torch.Tensor, x: torch.Tensor, num_bits: int = 16):
        """Apply Hilbert curve ordering."""
        bs, num_points, dim = x.shape

        # Normalize coordinates
        xyz_min = xyz.min(dim=1, keepdim=True)[0]
        xyz_max = xyz.max(dim=1, keepdim=True)[0]
        xyz_norm = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8) * 0.999

        # Quantize to integers
        xyz_int = (xyz_norm * (2 ** num_bits - 1)).long()
        xyz_int_flat = xyz_int.view(-1, 3)

        # Hilbert encoding
        hilbert_code = encode(xyz_int_flat, num_dims=3, num_bits=num_bits)
        hilbert_code = hilbert_code.view(bs, num_points)

        # Sort indices
        sort_idx = torch.argsort(hilbert_code, dim=1)
        unsort_idx = torch.argsort(sort_idx, dim=1)

        # Apply sort
        idx_expand = sort_idx.unsqueeze(-1).expand(-1, -1, dim)
        x_sorted = torch.gather(x, 1, idx_expand)

        return x_sorted, unsort_idx

    def _z_order_sort(self, xyz: torch.Tensor, x: torch.Tensor, num_bits: int = 10):
        """Apply Z-order (Morton) curve ordering."""
        bs, num_points, dim = x.shape

        # Normalize coordinates
        xyz_min = xyz.min(dim=1, keepdim=True)[0]
        xyz_max = xyz.max(dim=1, keepdim=True)[0]
        xyz_norm = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8)

        # Quantize
        xyz_int = (xyz_norm * (2 ** num_bits - 1)).long()

        # Morton encoding (interleave bits)
        def spread_bits(v, num_bits):
            """Spread bits for Morton code."""
            v = v & ((1 << num_bits) - 1)
            v = (v | (v << 16)) & 0x030000FF
            v = (v | (v << 8)) & 0x0300F00F
            v = (v | (v << 4)) & 0x030C30C3
            v = (v | (v << 2)) & 0x09249249
            return v

        morton_code = (
            spread_bits(xyz_int[..., 0], num_bits) |
            (spread_bits(xyz_int[..., 1], num_bits) << 1) |
            (spread_bits(xyz_int[..., 2], num_bits) << 2)
        )

        # Sort indices
        sort_idx = torch.argsort(morton_code, dim=1)
        unsort_idx = torch.argsort(sort_idx, dim=1)

        # Apply sort
        idx_expand = sort_idx.unsqueeze(-1).expand(-1, -1, dim)
        x_sorted = torch.gather(x, 1, idx_expand)

        return x_sorted, unsort_idx

    def forward(self, x: torch.Tensor, xyz: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - input features
            xyz: [B, N, 3] - point coordinates
            cond: [B, C] - condition features
        Returns:
            [B, N, D] - output features
        """
        bs, num_points, dim = x.shape

        # === Scan 1: Hilbert curve (forward + backward) ===
        x_hilbert, unsort_hilbert = self._hilbert_sort(xyz, x)

        for layer in self.layers:
            # Forward scan
            out_fwd = layer(x_hilbert, cond)
            # Backward scan
            out_bwd = layer(torch.flip(x_hilbert, dims=[1]), cond)
            out_bwd = torch.flip(out_bwd, dims=[1])
            # Residual
            x_hilbert = x_hilbert + out_fwd + out_bwd

        # Unsort back to original order
        unsort_expand = unsort_hilbert.unsqueeze(-1).expand(-1, -1, dim)
        x_hilbert_out = torch.gather(x_hilbert, 1, unsort_expand)

        # === Scan 2: Z-order curve (forward + backward) ===
        x_zorder, unsort_zorder = self._z_order_sort(xyz, x)

        for layer in self.layers:  # Shared weights
            out_fwd = layer(x_zorder, cond)
            out_bwd = layer(torch.flip(x_zorder, dims=[1]), cond)
            out_bwd = torch.flip(out_bwd, dims=[1])
            x_zorder = x_zorder + out_fwd + out_bwd

        unsort_expand = unsort_zorder.unsqueeze(-1).expand(-1, -1, dim)
        x_zorder_out = torch.gather(x_zorder, 1, unsort_expand)

        # === Fuse multi-scan features ===
        x_fused = self.scan_fusion(torch.cat([x_hilbert_out, x_zorder_out], dim=-1))

        return x_fused


class ContactHybridMamba(nn.Module):
    """
    Hybrid architecture combining:
    1. Multi-scan Mamba for local feature extraction (high contact precision)
    2. Perceiver-style bottleneck for global consistency (better distance field)
    3. FiLM-based language conditioning

    This architecture aims to surpass Perceiver by combining Mamba's local precision
    with Perceiver's global aggregation capability.
    """

    def __init__(self, arch_cfg, contact_dim: int, point_feat_dim: int, text_feat_dim: int, time_emb_dim: int):
        super().__init__()

        # Config
        self.trans_dim = getattr(arch_cfg, 'trans_dim', 256)
        self.depth = getattr(arch_cfg, 'depth', 4)
        self.last_dim = getattr(arch_cfg, 'last_dim', 256)
        self.num_latents = getattr(arch_cfg, 'num_latents', 128)
        self.num_heads = getattr(arch_cfg, 'num_heads', 8)
        self.use_multi_scan = getattr(arch_cfg, 'use_multi_scan', True)
        self.use_perceiver = getattr(arch_cfg, 'use_perceiver', True)

        # Input dimensions
        self.in_channels = contact_dim + point_feat_dim + 3  # +3 for xyz
        self.cond_dim = text_feat_dim + time_emb_dim

        # === Feature Embedding ===
        self.embedding = nn.Sequential(
            nn.Linear(self.in_channels, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.LayerNorm(self.trans_dim)
        )

        # === Condition Projection ===
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim)
        )

        # === Multi-Scan Mamba Encoder (Local Features) ===
        if self.use_multi_scan:
            self.mamba_encoder = MultiScanMamba(
                dim=self.trans_dim,
                cond_dim=self.trans_dim,
                depth=self.depth,
                num_scans=2
            )
        else:
            # Fallback to single-scan
            self.mamba_layers = nn.ModuleList([
                ConditionedMambaBlock(self.trans_dim, self.trans_dim)
                for _ in range(self.depth)
            ])

        # === Perceiver Bottleneck (Global Consistency) ===
        if self.use_perceiver:
            self.perceiver = PerceiverBottleneck(
                point_dim=self.trans_dim,
                latent_dim=self.trans_dim,
                num_latents=self.num_latents,
                num_heads=self.num_heads,
                num_self_attn_layers=2,
                dropout=0.1
            )

        # === Output Projection ===
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.trans_dim),
            nn.Linear(self.trans_dim, self.last_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        point_feat: Optional[torch.Tensor],
        language_feat: torch.Tensor,
        time_embedding: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of ContactHybridMamba.

        Args:
            x: input contact map, [bs, num_points, contact_dim]
            point_feat: [bs, num_points, point_feat_dim] or None
            language_feat: [bs, 1, text_feat_dim]
            time_embedding: [bs, 1, time_emb_dim]

        Returns:
            Output features, [bs, num_points, last_dim]
        """
        bs, num_points, _ = x.shape
        xyz = kwargs['c_pc_xyz']  # [B, N, 3]

        # === 1. Feature Fusion ===
        features_list = [x, xyz]
        if point_feat is not None:
            features_list.append(point_feat)
        fusion_feat = torch.cat(features_list, dim=-1)  # [B, N, C_in]

        # Embedding
        x_emb = self.embedding(fusion_feat)  # [B, N, trans_dim]

        # === 2. Condition Preparation ===
        cond = torch.cat([
            language_feat.squeeze(1),  # [B, text_dim]
            time_embedding.squeeze(1)  # [B, time_dim]
        ], dim=-1)  # [B, cond_dim]
        cond = self.cond_proj(cond)  # [B, trans_dim]

        # === 3. Multi-Scan Mamba Encoder (Local Features) ===
        if self.use_multi_scan:
            x_local = self.mamba_encoder(x_emb, xyz, cond)
        else:
            # Single-scan fallback with Hilbert ordering
            x_sorted, unsort_idx = self._hilbert_sort(xyz, x_emb)
            for layer in self.mamba_layers:
                out_fwd = layer(x_sorted, cond)
                out_bwd = layer(torch.flip(x_sorted, dims=[1]), cond)
                out_bwd = torch.flip(out_bwd, dims=[1])
                x_sorted = x_sorted + out_fwd + out_bwd
            unsort_expand = unsort_idx.unsqueeze(-1).expand(-1, -1, self.trans_dim)
            x_local = torch.gather(x_sorted, 1, unsort_expand)

        # === 4. Perceiver Bottleneck (Global Consistency) ===
        if self.use_perceiver:
            x_global = self.perceiver(x_local)
        else:
            x_global = x_local

        # === 5. Output ===
        return self.out_proj(x_global)

    def _hilbert_sort(self, xyz: torch.Tensor, x: torch.Tensor, num_bits: int = 16):
        """Hilbert curve sorting for single-scan fallback."""
        bs, num_points, dim = x.shape

        xyz_min = xyz.min(dim=1, keepdim=True)[0]
        xyz_max = xyz.max(dim=1, keepdim=True)[0]
        xyz_norm = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8) * 0.999

        xyz_int = (xyz_norm * (2 ** num_bits - 1)).long()
        xyz_int_flat = xyz_int.view(-1, 3)
        hilbert_code = encode(xyz_int_flat, num_dims=3, num_bits=num_bits)
        hilbert_code = hilbert_code.view(bs, num_points)

        sort_idx = torch.argsort(hilbert_code, dim=1)
        unsort_idx = torch.argsort(sort_idx, dim=1)

        idx_expand = sort_idx.unsqueeze(-1).expand(-1, -1, dim)
        x_sorted = torch.gather(x, 1, idx_expand)

        return x_sorted, unsort_idx
