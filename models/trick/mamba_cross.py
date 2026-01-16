import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat


class _SelectiveStateSpace(nn.Module):
    """轻量版 Mamba Mixer，遵循官方 selective scan 接口。"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias)

        self.conv_x = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        # x: [B, L, d_model]
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z_inner = xz.chunk(2, dim=-1)

        # depthwise conv
        x_inner = x_inner.transpose(1, 2)  # [B, d_inner, L]
        x_inner = self.conv_x(x_inner)[..., :L]
        x_inner = F.silu(x_inner).transpose(1, 2)  # [B, L, d_inner]

        params = self.x_proj(x_inner)
        dt, B_ssm, C_ssm = torch.split(params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).transpose(1, 2).contiguous()  # [B, d_inner, L]
        B_ssm = B_ssm.transpose(1, 2).contiguous()
        C_ssm = C_ssm.transpose(1, 2).contiguous()
        A = -torch.exp(self.A_log.float())

        y_ssm = selective_scan_fn(
            x_inner.transpose(1, 2).contiguous(),  # [B, d_inner, L]
            dt,
            A,
            B_ssm,
            C_ssm,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y_ssm = y_ssm.transpose(1, 2)  # [B, L, d_inner]

        z_inner = F.silu(z_inner)
        y = torch.cat([y_ssm, z_inner], dim=-1)
        return self.out_proj(y)


class BidirectionalMambaBlock(nn.Module):
    """
    [方案 B] 双向 Mamba + Cross Attention 混合模块
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Norm
        self.norm1 = nn.LayerNorm(d_model)

        # 2. 双向 Mixers (权重不共享)
        self.mixer_forward = _SelectiveStateSpace(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mixer_backward = _SelectiveStateSpace(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # 3. 融合层
        self.fusion_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.fusion_drop = nn.Dropout(drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # =====================================================================
        # [新增] Cross Attention 模块 (Query=Motion, Key/Value=Text)
        # =====================================================================
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,  # 默认8头，足够处理 512 维度
            dropout=drop,
            batch_first=True
        )
        # =====================================================================

        # 4. FFN 部分
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=int(d_model * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop
        )

    def forward(self, x, padding_mask=None, context=None, context_mask=None):
        """
        Args:
            x: [B, L, C] 输入序列 (Motion)
            padding_mask: [B, L] True 表示是 padding (需要被 mask 掉)
            context: [B, S, C] 上下文序列 (Text Embeddings)
            context_mask: [B, S] 上下文 Mask (True 表示 padding)
        """
        # --- Part 1: Bidirectional Mamba (自建模) ---
        residual = x
        h = self.norm1(x)

        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Forward
        h_fwd = self.mixer_forward(h)
        # Backward
        h_bwd = torch.flip(h, dims=[1])
        h_bwd = self.mixer_backward(h_bwd)
        h_bwd = torch.flip(h_bwd, dims=[1])

        # Fusion
        h_fused = torch.cat([h_fwd, h_bwd], dim=-1)
        h_fused = self.fusion_proj(h_fused)
        h_fused = self.fusion_drop(h_fused)

        # Residual 1
        x = residual + self.drop_path(h_fused)

        # --- Part 2: [新增] Cross Attention (语义对齐) ---
        if context is not None:
            residual = x
            h = self.norm_cross(x)

            # Query = Motion (x)
            # Key = Text (context)
            # Value = Text (context)
            h_attn, _ = self.cross_attn(
                query=h,
                key=context,
                value=context,
                key_padding_mask=context_mask
            )

            # Residual 2
            x = residual + self.drop_path(h_attn)

        # --- Part 3: FFN ---
        residual = x
        h = self.norm2(x)
        h = self.mlp(h)

        # Residual 3
        x = residual + self.drop_path(h)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x


class MambaBlock(nn.Module):
    # 保持原样，未修改
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = _SelectiveStateSpace(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(in_features=d_model, hidden_features=int(d_model * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x, padding_mask=None):
        residual = x
        h = self.norm1(x)
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        h = self.mixer(h)
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = residual + self.drop_path(h)

        residual = x
        h = self.norm2(x)
        h = self.mlp(h)
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = residual + self.drop_path(h)
        return x