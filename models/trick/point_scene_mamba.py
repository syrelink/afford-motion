import torch
import torch.nn as nn
from mamba_ssm import Mamba
# 假设 hilbert.py 在同级目录下
from .hilbert import encode


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
    