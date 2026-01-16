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


class MambaBlock(nn.Module):
    """带有 LayerNorm + FFN 的残差 Mamba 模块，兼容 padding_mask。"""

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
        # x: [B, L, C], padding_mask: [B, L]
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


class BidirectionalMambaAdaLNBlock(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
            text_dim=512,  # [新增] 假设 CLIP text embedding 是 512 维
    ):
        super().__init__()
        self.d_model = d_model

        # [关键修改 1] 改回 True (或者直接删掉 elementwise_affine 参数，默认就是 True)
        # 这样它就能加载旧模型里 norm1.weight 和 norm1.bias 了！
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True)

        # [新增] AdaLN 调制层
        # 输入: 全局文本特征 -> 输出: 2 * d_model (Scale + Shift)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(text_dim, 2 * d_model, bias=True)
        )
        # 零初始化：保证训练初期和普通 Norm 一模一样，不破坏 FID
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # 2. 双向 Mixers
        self.mixer_forward = _SelectiveStateSpace(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mixer_backward = _SelectiveStateSpace(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # 3. 融合层
        self.fusion_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.fusion_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 4. FFN 部分 (Norm2 通常保持原样，或者也用 AdaLN，这里只改 Norm1 够用了)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=int(d_model * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop
        )

    def forward(self, x, padding_mask=None, context=None):
        """
        Args:
            x: [B, L, C] 动作序列
            context: [B, Text_Dim] 全局文本特征 (注意：必须是 Pooling 后的向量，不是序列)
        """
        residual = x

        # --- [AdaLN 逻辑开始] ---
        # [关键修改 2] 前向传播逻辑微调
        # 1. 先过带旧参数的 Norm (恢复旧模型行为)
        x_norm = self.norm1(x)
        # 1. 计算调制参数
        # shift, scale: [B, C]
        shift, scale = self.adaLN_modulation(context).chunk(2, dim=1)

        # 2. 应用调制
        # x 经过无参数 Norm 后，被 scale 缩放，被 shift 偏移
        # 公式：(旧Norm输出) * (1 + 0) + 0 = 旧Norm输出
        # 这样一开始 AdaLN 是透明的，完全保留了预训练效果
        h = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        # --- [AdaLN 逻辑结束] ---

        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # --- 双向 Mamba (保持不变) ---
        h_fwd = self.mixer_forward(h)
        h_bwd = torch.flip(h, dims=[1])
        h_bwd = self.mixer_backward(h_bwd)
        h_bwd = torch.flip(h_bwd, dims=[1])

        h_fused = torch.cat([h_fwd, h_bwd], dim=-1)
        h_fused = self.fusion_proj(h_fused)
        h_fused = self.fusion_drop(h_fused)

        x = residual + self.drop_path(h_fused)

        # --- FFN ---
        residual = x
        h = self.norm2(x)
        h = self.mlp(h)
        x = residual + self.drop_path(h)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x