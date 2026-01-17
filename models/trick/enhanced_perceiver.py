import torch
import torch.nn as nn
from omegaconf import DictConfig
from models.modules import CrossAttentionLayer, SelfAttentionBlock


class EnhancedTextEncoder(nn.Module):
    """增强的文本编码器"""

    def __init__(self, text_feat_dim: int, trans_dim: int, dropout: float = 0.1):
        super().__init__()

        # 多层文本适配器
        self.text_adapter = nn.Sequential(
            nn.Linear(text_feat_dim, trans_dim),
            nn.LayerNorm(trans_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trans_dim, trans_dim),
            nn.LayerNorm(trans_dim),
        )

        # 文本增强（训练时使用）
        self.text_augmentation = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(trans_dim, trans_dim),
            nn.GELU(),
        )

        # 文本-场景对齐层
        self.scene_alignment = nn.MultiheadAttention(
            embed_dim=trans_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, text_emb: torch.Tensor, scene_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            text_emb: [B, 1, text_feat_dim]
            scene_emb: [B, N, trans_dim] 可选，用于对齐

        Returns:
            [B, 1, trans_dim]
        """
        # 1. 适配文本特征
        text_feat = self.text_adapter(text_emb)

        # 2. 文本增强（训练时）
        if self.training:
            text_feat = self.text_augmentation(text_feat)

        # 3. 文本-场景对齐（如果有场景特征）
        if scene_emb is not None:
            aligned_text, _ = self.scene_alignment(
                text_feat, scene_emb, scene_emb
            )
            return aligned_text

        return text_feat


class EnhancedPointEncoder(nn.Module):
    """增强的点云编码器"""

    def __init__(self, contact_dim: int, point_feat_dim: int, trans_dim: int,
                 num_neighbors: int = 16, dropout: float = 0.1):
        super().__init__()

        self.num_neighbors = num_neighbors

        # 多尺度点云编码
        self.local_encoder = nn.Sequential(
            nn.Linear(contact_dim + point_feat_dim + 3, trans_dim),
            nn.LayerNorm(trans_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.region_encoder = nn.Sequential(
            nn.Linear(trans_dim, trans_dim),
            nn.LayerNorm(trans_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trans_dim, trans_dim),
            nn.LayerNorm(trans_dim),
        )

        self.global_encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(trans_dim, trans_dim),
            nn.LayerNorm(trans_dim),
        )

        # 点云注意力
        self.point_attention = nn.MultiheadAttention(
            embed_dim=trans_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # 空间位置编码
        self.spatial_encoding = nn.Sequential(
            nn.Linear(6, trans_dim),  # 相对坐标 + 距离
            nn.LayerNorm(trans_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, point_feat: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, contact_dim]
            point_feat: [B, N, point_feat_dim]
            xyz: [B, N, 3]

        Returns:
            [B, N, trans_dim]
        """
        bs, num_points, _ = x.shape

        # 1. 局部特征
        local_feat = self.local_encoder(torch.cat([x, point_feat, xyz], dim=-1))

        # 2. 区域特征（邻居聚合）
        dist = torch.cdist(xyz, xyz)
        _, neighbor_idx = torch.topk(dist, self.num_neighbors, dim=2, largest=False)

        region_features = []
        for i in range(num_points):
            neighbors = local_feat[:, neighbor_idx[:, i, :], :]
            region_feat = neighbors.mean(dim=1, keepdim=True)
            region_features.append(region_feat)

        region_feat = torch.cat(region_features, dim=1)
        region_feat = self.region_encoder(region_feat)

        # 3. 全局特征
        global_feat = self.global_encoder(local_feat.transpose(1, 2)).transpose(1, 2)
        global_feat = global_feat.repeat(1, num_points, 1)

        # 4. 点云注意力
        combined = local_feat + region_feat + global_feat
        attended, _ = self.point_attention(combined, combined, combined)

        # 5. 空间位置编码
        rel_xyz = xyz.unsqueeze(2) - xyz.unsqueeze(1)
        rel_dist = torch.norm(rel_xyz, dim=-1, keepdim=True)
        spatial_feat = self.spatial_encoding(torch.cat([rel_xyz, rel_dist], dim=-1))

        return attended + spatial_feat


class EnhancedCrossAttention(nn.Module):
    """增强的跨注意力机制"""

    def __init__(self, trans_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        # 多层跨注意力
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(
                num_heads=8,
                num_q_input_channels=trans_dim,
                num_kv_input_channels=trans_dim,
                widening_factor=2,
                dropout=dropout,
                residual_dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 门控机制
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trans_dim, trans_dim),
                nn.LayerNorm(trans_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(trans_dim, trans_dim),
                nn.Sigmoid(),
            )
            for _ in range(num_layers)
        ])

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, Q, trans_dim]
            key_value: [B, KV, trans_dim]

        Returns:
            [B, Q, trans_dim]
        """
        for cross_attn, gate in zip(self.cross_attn_layers, self.gate_layers):
            # 跨注意力
            out = cross_attn(query, key_value)

            # 门控融合
            gate_weight = gate(query)
            query = query + gate_weight * out

        return query


class PhysicsConstraintLayer(nn.Module):
    """物理约束层"""

    def __init__(self, trans_dim: int, dropout: float = 0.1):
        super().__init__()

        # 接触概率约束
        self.contact_prob = nn.Sequential(
            nn.Linear(trans_dim, trans_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trans_dim, 1),
            nn.Sigmoid(),
        )

        # 重力约束（可学习的偏置）
        self.gravity_bias = nn.Parameter(torch.zeros(1, 1, trans_dim))

        # 支撑面检测
        self.support_detector = nn.Sequential(
            nn.Linear(trans_dim + 3, trans_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trans_dim, 1),
            nn.Sigmoid(),
        )

        # 接触类型分类
        self.contact_type = nn.Sequential(
            nn.Linear(trans_dim, 4),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> dict:
        """
        Args:
            x: [B, N, trans_dim]
            xyz: [B, N, 3]

        Returns:
            dict with constrained features and auxiliary outputs
        """
        bs, num_points, _ = x.shape

        # 1. 接触概率
        contact_prob = self.contact_prob(x)

        # 2. 重力约束：底部点更可能接触
        y_coords = xyz[:, :, 1:2]
        y_min = y_coords.min(dim=1, keepdim=True)[0]
        y_max = y_coords.max(dim=1, keepdim=True)[0]
        gravity_weight = 1.0 - (y_coords - y_min) / (y_max - y_min + 1e-8)

        # 3. 支撑面检测
        support_feat = torch.cat([x, xyz], dim=-1)
        support_prob = self.support_detector(support_feat)

        # 4. 接触类型
        contact_type = self.contact_type(x)

        # 5. 物理约束融合
        constrained = x * contact_prob * gravity_weight * support_prob

        return {
            'features': constrained,
            'contact_prob': contact_prob,
            'contact_type': contact_type,
            'gravity_weight': gravity_weight,
        }


class MultiTaskContactPerceiver(nn.Module):
    """多任务接触点 Perceiver"""

    def __init__(self, arch_cfg: DictConfig, contact_dim: int, point_feat_dim: int,
                 text_feat_dim: int, time_emb_dim: int):
        super().__init__()

        self.trans_dim = getattr(arch_cfg, 'trans_dim', 256)
        self.last_dim = getattr(arch_cfg, 'last_dim', 256)

        # 1. 增强文本编码
        self.text_encoder = EnhancedTextEncoder(text_feat_dim, self.trans_dim)

        # 2. 增强点云编码
        self.point_encoder = EnhancedPointEncoder(
            contact_dim, point_feat_dim, self.trans_dim
        )

        # 3. 增强跨注意力
        self.cross_attention = EnhancedCrossAttention(self.trans_dim)

        # 4. 物理约束
        self.physics_constraint = PhysicsConstraintLayer(self.trans_dim)

        # 5. 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.LayerNorm(self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.last_dim),
        )

        # 6. 多任务头
        self.contact_head = nn.Sequential(
            nn.Linear(self.last_dim, contact_dim),
            nn.Sigmoid(),
        )

        self.contact_type_head = nn.Sequential(
            nn.Linear(self.last_dim, 4),
            nn.Softmax(dim=-1),
        )

        self.force_direction_head = nn.Sequential(
            nn.Linear(self.last_dim, 3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, point_feat: torch.Tensor,
                language_feat: torch.Tensor, time_embedding: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, N, contact_dim]
            point_feat: [B, N, point_feat_dim] or None
            language_feat: [B, 1, text_feat_dim]
            time_embedding: [B, 1, time_emb_dim]
            **kwargs: c_pc_xyz [B, N, 3]

        Returns:
            [B, N, contact_dim]
        """
        bs, num_points, _ = x.shape
        xyz = kwargs['c_pc_xyz']

        # 1. 文本编码
        text_emb = self.text_encoder(language_feat)

        # 2. 点云编码
        point_emb = self.point_encoder(x, point_feat, xyz)

        # 3. 跨注意力融合
        # 文本作为 query，点云作为 key/value
        fused = self.cross_attention(text_emb, point_emb)

        # 4. 物理约束
        constrained = self.physics_constraint(fused, xyz)

        # 5. 输出投影
        features = self.output_proj(constrained['features'])

        # 6. 多任务输出
        contact_map = self.contact_head(features)

        return contact_map


class EnhancedContactPerceiver(nn.Module):
    """增强的 ContactPerceiver（兼容原始接口）"""

    def __init__(self, arch_cfg: DictConfig, contact_dim: int, point_feat_dim: int,
                 text_feat_dim: int, time_emb_dim: int):
        super().__init__()

        self.trans_dim = getattr(arch_cfg, 'trans_dim', 256)
        self.last_dim = getattr(arch_cfg, 'last_dim', 256)

        # 1. 增强文本编码
        self.text_encoder = EnhancedTextEncoder(text_feat_dim, self.trans_dim)

        # 2. 增强点云编码
        self.point_encoder = EnhancedPointEncoder(
            contact_dim, point_feat_dim, self.trans_dim
        )

        # 3. 增强跨注意力
        self.cross_attention = EnhancedCrossAttention(self.trans_dim)

        # 4. 物理约束
        self.physics_constraint = PhysicsConstraintLayer(self.trans_dim)

        # 5. 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.LayerNorm(self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.last_dim),
        )

    def forward(self, x: torch.Tensor, point_feat: torch.Tensor,
                language_feat: torch.Tensor, time_embedding: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, N, contact_dim]
            point_feat: [B, N, point_feat_dim] or None
            language_feat: [B, 1, text_feat_dim]
            time_embedding: [B, 1, time_emb_dim]
            **kwargs: c_pc_xyz [B, N, 3]

        Returns:
            [B, N, last_dim]
        """
        bs, num_points, _ = x.shape
        xyz = kwargs['c_pc_xyz']

        # 1. 文本编码
        text_emb = self.text_encoder(language_feat)

        # 2. 点云编码
        point_emb = self.point_encoder(x, point_feat, xyz)

        # 3. 跨注意力融合
        # 文本作为 query，点云作为 key/value
        fused = self.cross_attention(text_emb, point_emb)

        # 4. 物理约束
        constrained = self.physics_constraint(fused, xyz)

        # 5. 输出投影
        features = self.output_proj(constrained['features'])

        return features
