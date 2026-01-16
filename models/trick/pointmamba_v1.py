import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba
from ..pointmamba.hilbert import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm import Mamba
from .hilbert import encode  # 确保 hilbert.py 在同级目录


# ==========================================
# 1. 新的核心模块: BiMamba + AdaLN
# ==========================================
class BiMambaBlockAdaLN(nn.Module):
    def __init__(self, dim, d_state=16, expand=2, dropout=0.1):
        super().__init__()

        # --- 双向 Mamba ---
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, expand=expand)

        self.dropout = nn.Dropout(dropout)

        # --- Feed Forward ---
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

        # --- AdaLN (自适应归一化) ---
        self.silu = nn.SiLU()
        # 不带参数的 LayerNorm，因为参数由 cond 生成
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # 调制层：输入 cond，输出 4 组参数 (shift_m, scale_m, shift_f, scale_f)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 4 * dim, bias=True)
        )

        # Zero-Init 技巧：让训练初期 Block 近似恒等映射
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, cond):
        """
        x: [B, N, C]
        cond: [B, C] (维度必须与 x 的 C 一致)
        """
        # 1. 计算调制参数
        # shift, scale: [B, C] -> 广播到 [B, 1, C]
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(cond).chunk(4, dim=1)
        shift_msa, scale_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)
        shift_mlp, scale_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1)

        # 2. Mamba 混合层 (Bi-Directional)
        res = x
        # AdaLN: modulate(norm(x))
        x = self.norm1(x) * (1 + scale_msa) + shift_msa

        # 双向扫描
        x_fwd = self.mamba_fwd(x)
        # 反向：翻转输入 -> Mamba -> 翻转回来
        x_bwd = self.mamba_bwd(x.flip([1])).flip([1])
        x = x_fwd + x_bwd  # 融合

        x = self.dropout(x)
        x = x + res

        # 3. FFN 层
        res = x
        x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = self.ffn(x)
        x = x + res

        return x


# ==========================================
# 2. 辅助模块 (保持不变)
# ==========================================
class PointDownsample(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=0.25):
        super().__init__()
        self.ratio = ratio
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, xyz):
        B, N, C = x.shape
        target_N = int(N * self.ratio)
        # 随机采样 (配合 Hilbert 排序使用效果也不错)
        idx = torch.randperm(N, device=x.device)[:target_N].sort()[0]
        new_x = x[:, idx, :]
        new_xyz = xyz[:, idx, :]
        new_x = self.proj(new_x)
        return new_x, new_xyz


class PointUpsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, x_skip, target_len):
        x = x.transpose(1, 2)
        x_upsampled = F.interpolate(x, size=target_len, mode='nearest')
        x_upsampled = x_upsampled.transpose(1, 2)
        x_feat = self.proj(x_upsampled)
        out = torch.cat([x_feat, x_skip], dim=-1)
        return out


# ==========================================
# 3. 主模型: HighResPointMambaUNet (升级版)
# ==========================================
class HighResPointMambaUNet(nn.Module):
    def __init__(self, arch_cfg, contact_dim, point_feat_dim, text_feat_dim, time_emb_dim):
        super().__init__()

        self.dim = getattr(arch_cfg, 'base_dim', 128)
        self.input_dim = contact_dim + point_feat_dim + 3

        # 文本/时间映射
        self.cond_proj = nn.Linear(text_feat_dim + time_emb_dim, self.dim)

        # --- Encoder ---
        self.in_proj = nn.Linear(self.input_dim, self.dim)

        # 使用新定义的 BiMambaBlockAdaLN
        Block = BiMambaBlockAdaLN

        # Stage 1 (Full Res, dim=128)
        self.enc1 = Block(self.dim)
        self.down1 = PointDownsample(self.dim, self.dim * 2, ratio=0.25)

        # Stage 2 (1/4 Res, dim=256)
        self.enc2 = Block(self.dim * 2)
        self.down2 = PointDownsample(self.dim * 2, self.dim * 4, ratio=0.25)

        # --- Bottleneck (1/16 Res, dim=512) ---
        self.mid_mamba = Block(self.dim * 4)

        # --- Decoder ---
        # 注意：因为要传 cond 参数，Sequential 不好用了，我们拆开写

        # Stage 2 Upsample (512 -> 256)
        self.up2 = PointUpsample(self.dim * 4, self.dim * 2)
        self.dec2_proj = nn.Linear(self.dim * 4, self.dim * 2)  # 融合后的维度调整
        self.dec2_mamba = Block(self.dim * 2)

        # Stage 1 Upsample (256 -> 128)
        self.up1 = PointUpsample(self.dim * 2, self.dim)
        self.dec1_proj = nn.Linear(self.dim * 2, self.dim)
        self.dec1_mamba = Block(self.dim)

        # 输出头：输出 128 维特征，交给 CDM 做投影
        self.out_head = nn.LayerNorm(self.dim)

    def reorder_points_hilbert(self, xyz, features):
        """ Hilbert 排序 (保持不变) """
        B, N, _ = xyz.shape
        min_xyz = xyz.min(dim=1, keepdim=True)[0]
        max_xyz = xyz.max(dim=1, keepdim=True)[0]
        norm_xyz = (xyz - min_xyz) / (max_xyz - min_xyz + 1e-6)

        num_bits = 16
        scale = (1 << num_bits) - 1
        int_xyz = (norm_xyz * scale).long()
        int_xyz_flat = int_xyz.view(-1, 3)

        hilbert_codes = encode(int_xyz_flat, num_dims=3, num_bits=num_bits)
        hilbert_codes = hilbert_codes.view(B, N)
        sorted_idx = torch.argsort(hilbert_codes, dim=-1)

        batch_idx = torch.arange(B, device=xyz.device).unsqueeze(1).expand(-1, N)
        xyz_sorted = xyz[batch_idx, sorted_idx, :]
        features_sorted = features[batch_idx, sorted_idx, :]
        restore_idx = torch.argsort(sorted_idx, dim=-1)

        return xyz_sorted, features_sorted, restore_idx

    def forward(self, x, point_feat, language_feat, time_embedding, **kwargs):
        # 1. 数据准备
        xyz = kwargs['c_pc_xyz']
        original_N = xyz.shape[1]

        if point_feat is not None:
            x = torch.cat([x, point_feat], dim=-1)

        # 2. Hilbert 排序
        # [关键] 先排序，让 Mamba 能读懂几何
        x_sorted, features_sorted, restore_idx = self.reorder_points_hilbert(xyz, x)
        x_input = torch.cat([features_sorted, x_sorted], dim=-1)  # [B, N, C+3]

        # 3. 准备条件特征 (Condition)
        # [关键] AdaLN 需要这里的 cond
        cond = torch.cat([language_feat, time_embedding], dim=-1)
        cond_feat = self.cond_proj(cond).squeeze(1)  # [B, dim]

        # 为不同层级准备不同维度的 condition
        # dim=128
        cond_128 = cond_feat
        # dim=256 (简单的 Padding，或者你可以额外定义 Linear 层来映射)
        cond_256 = F.pad(cond_feat, (0, self.dim))
        # dim=512
        cond_512 = F.pad(cond_feat, (0, self.dim * 3))

        # 4. 网络前向传播
        # --- Encoder ---
        x0 = self.in_proj(x_input)  # [B, N, 128]

        # Level 1
        x1_skip = self.enc1(x0, cond_128)  # 传入 cond

        # Downsample 1
        x1, xyz1 = self.down1(x1_skip, x_sorted)

        # Level 2
        x2_skip = self.enc2(x1, cond_256)  # 传入 cond (维度匹配)

        # Downsample 2
        x2, xyz2 = self.down2(x2_skip, xyz1)

        # --- Bottleneck ---
        x_mid = self.mid_mamba(x2, cond_512)  # 传入 cond

        # --- Decoder ---
        # Up to Stage 2
        # Upsample: 512 -> 512 (插值后) + 256 (Skip) = 768
        x_up2 = self.up2(x_mid, x2_skip, target_len=x1.shape[1])
        x_up2 = self.dec2_proj(x_up2)  # 768 -> 256
        x_dec2 = self.dec2_mamba(x_up2, cond_256)  # Mamba Refine

        # Up to Stage 1
        # Upsample: 256 -> 256 (插值后) + 128 (Skip) = 384
        x_up1 = self.up1(x_dec2, x1_skip, target_len=original_N)
        x_up1 = self.dec1_proj(x_up1)  # 384 -> 128
        x_final = self.dec1_mamba(x_up1, cond_128)  # Mamba Refine

        # 5. 输出特征
        out_sorted = self.out_head(x_final)  # [B, N, 128]

        # 6. 恢复原始顺序
        B, N, _ = out_sorted.shape
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, N)
        out_restored = out_sorted[batch_idx, restore_idx, :]

        return out_restored