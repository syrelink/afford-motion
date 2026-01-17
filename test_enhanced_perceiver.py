#!/usr/bin/env python3
"""
测试增强 Perceiver 的兼容性和功能
"""

import torch
from omegaconf import DictConfig
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.cdm import CDM
from models.trick.enhanced_perceiver import (
    EnhancedContactPerceiver,
    EnhancedTextEncoder,
    EnhancedPointEncoder,
    EnhancedCrossAttention,
    PhysicsConstraintLayer,
)


def test_enhanced_components():
    """测试增强组件"""
    print("=" * 60)
    print("测试增强组件")
    print("=" * 60)

    # 测试配置
    trans_dim = 256
    contact_dim = 2
    point_feat_dim = 256
    text_feat_dim = 512
    time_emb_dim = 256
    num_points = 1024

    # 1. 测试 EnhancedTextEncoder
    print("\n1. 测试 EnhancedTextEncoder...")
    text_encoder = EnhancedTextEncoder(text_feat_dim, trans_dim)
    text_emb = torch.randn(2, 1, text_feat_dim)
    scene_emb = torch.randn(2, num_points, trans_dim)
    text_out = text_encoder(text_emb, scene_emb)
    print(f"   输入: {text_emb.shape}, 输出: {text_out.shape}")
    assert text_out.shape == (2, 1, trans_dim), "输出形状错误"
    print("   ✅ EnhancedTextEncoder 测试通过")

    # 2. 测试 EnhancedPointEncoder
    print("\n2. 测试 EnhancedPointEncoder...")
    point_encoder = EnhancedPointEncoder(contact_dim, point_feat_dim, trans_dim)
    x = torch.randn(2, num_points, contact_dim)
    point_feat = torch.randn(2, num_points, point_feat_dim)
    xyz = torch.randn(2, num_points, 3)
    point_out = point_encoder(x, point_feat, xyz)
    print(f"   输入: x={x.shape}, point_feat={point_feat.shape}, xyz={xyz.shape}")
    print(f"   输出: {point_out.shape}")
    assert point_out.shape == (2, num_points, trans_dim), "输出形状错误"
    print("   ✅ EnhancedPointEncoder 测试通过")

    # 3. 测试 EnhancedCrossAttention
    print("\n3. 测试 EnhancedCrossAttention...")
    cross_attn = EnhancedCrossAttention(trans_dim)
    query = torch.randn(2, 1, trans_dim)
    key_value = torch.randn(2, num_points, trans_dim)
    attn_out = cross_attn(query, key_value)
    print(f"   输入: query={query.shape}, key_value={key_value.shape}")
    print(f"   输出: {attn_out.shape}")
    assert attn_out.shape == (2, 1, trans_dim), "输出形状错误"
    print("   ✅ EnhancedCrossAttention 测试通过")

    # 4. 测试 PhysicsConstraintLayer
    print("\n4. 测试 PhysicsConstraintLayer...")
    physics_constraint = PhysicsConstraintLayer(trans_dim)
    features = torch.randn(2, num_points, trans_dim)
    xyz = torch.randn(2, num_points, 3)
    constrained = physics_constraint(features, xyz)
    print(f"   输入: features={features.shape}, xyz={xyz.shape}")
    print(f"   输出: features={constrained['features'].shape}")
    print(f"   接触概率: {constrained['contact_prob'].shape}")
    print(f"   接触类型: {constrained['contact_type'].shape}")
    assert constrained['features'].shape == (2, num_points, trans_dim), "输出形状错误"
    assert constrained['contact_prob'].shape == (2, num_points, 1), "接触概率形状错误"
    assert constrained['contact_type'].shape == (2, num_points, 4), "接触类型形状错误"
    print("   ✅ PhysicsConstraintLayer 测试通过")

    # 5. 测试 EnhancedContactPerceiver
    print("\n5. 测试 EnhancedContactPerceiver...")
    arch_cfg = DictConfig({
        'trans_dim': trans_dim,
        'last_dim': 256,
        'num_neighbors': 16,
        'dropout': 0.1,
    })
    perceiver = EnhancedContactPerceiver(
        arch_cfg, contact_dim, point_feat_dim, text_feat_dim, time_emb_dim
    )
    x = torch.randn(2, num_points, contact_dim)
    point_feat = torch.randn(2, num_points, point_feat_dim)
    language_feat = torch.randn(2, 1, text_feat_dim)
    time_embedding = torch.randn(2, 1, time_emb_dim)
    c_pc_xyz = torch.randn(2, num_points, 3)
    perceiver_out = perceiver(x, point_feat, language_feat, time_embedding, c_pc_xyz=c_pc_xyz)
    print(f"   输入: x={x.shape}, point_feat={point_feat.shape}")
    print(f"   输出: {perceiver_out.shape}")
    assert perceiver_out.shape == (2, num_points, arch_cfg.last_dim), "输出形状错误"
    print("   ✅ EnhancedContactPerceiver 测试通过")

    print("\n" + "=" * 60)
    print("所有增强组件测试通过！")
    print("=" * 60)


def test_cdm_integration():
    """测试 CDM 集成"""
    print("\n" + "=" * 60)
    print("测试 CDM 集成")
    print("=" * 60)

    # 测试配置
    cfg = DictConfig({
        'arch': 'EnhancedPerceiver',
        'arch_enhanced_perceiver': {
            'trans_dim': 256,
            'last_dim': 256,
            'num_neighbors': 16,
            'dropout': 0.1,
        },
        'data_repr': 'contact_map',
        'input_feats': 2,
        'time_emb_dim': 256,
        'text_model': {
            'version': 'clip-ViT-B/32',
            'max_length': 77,
        },
        'scene_model': {
            'use_scene_model': True,
            'name': 'pointtransformer',
            'point_feat_dim': 256,
            'num_points': 1024,
            'pretrained_weight': None,
            'freeze': True,
        },
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建 CDM 模型
    print("\n创建 CDM 模型...")
    model = CDM(cfg, device=device).to(device)

    # 打印模型信息
    print(f"\n模型架构: {cfg.arch}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 2
    num_points = 1024

    x = torch.randn(batch_size, num_points, 2).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    c_text = ["a person sitting on a chair"] * batch_size
    c_pc_xyz = torch.randn(batch_size, num_points, 3).to(device)
    c_pc_feat = torch.randn(batch_size, num_points, 256).to(device)

    with torch.no_grad():
        output = model(
            x=x,
            timesteps=timesteps,
            c_text=c_text,
            c_pc_xyz=c_pc_xyz,
            c_pc_feat=c_pc_feat,
        )

    print(f"输入形状: x={x.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == (batch_size, num_points, 2), "输出形状错误"
    print("   ✅ CDM 集成测试通过")

    print("\n" + "=" * 60)
    print("CDM 集成测试通过！")
    print("=" * 60)


def test_backward_compatibility():
    """测试向后兼容性（原始 Perceiver）"""
    print("\n" + "=" * 60)
    print("测试向后兼容性（原始 Perceiver）")
    print("=" * 60)

    # 测试原始 Perceiver
    cfg = DictConfig({
        'arch': 'Perceiver',
        'arch_perceiver': {
            'encoder_q_input_channels': 256,
            'encoder_kv_input_channels': 256,
            'encoder_num_heads': 8,
            'encoder_widening_factor': 2,
            'encoder_dropout': 0.1,
            'encoder_residual_dropout': 0.1,
            'encoder_self_attn_num_layers': 1,
            'decoder_q_input_channels': 256,
            'decoder_kv_input_channels': 256,
            'decoder_num_heads': 8,
            'decoder_widening_factor': 2,
            'decoder_dropout': 0.1,
            'decoder_residual_dropout': 0.1,
            'point_pos_emb': False,
            'last_dim': 256,
        },
        'data_repr': 'contact_map',
        'input_feats': 2,
        'time_emb_dim': 256,
        'text_model': {
            'version': 'clip-ViT-B/32',
            'max_length': 77,
        },
        'scene_model': {
            'use_scene_model': True,
            'name': 'pointtransformer',
            'point_feat_dim': 256,
            'num_points': 1024,
            'pretrained_weight': None,
            'freeze': True,
        },
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建 CDM 模型（原始 Perceiver）
    print("\n创建 CDM 模型（原始 Perceiver）...")
    model = CDM(cfg, device=device).to(device)

    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 2
    num_points = 1024

    x = torch.randn(batch_size, num_points, 2).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    c_text = ["a person sitting on a chair"] * batch_size
    c_pc_xyz = torch.randn(batch_size, num_points, 3).to(device)
    c_pc_feat = torch.randn(batch_size, num_points, 256).to(device)

    with torch.no_grad():
        output = model(
            x=x,
            timesteps=timesteps,
            c_text=c_text,
            c_pc_xyz=c_pc_xyz,
            c_pc_feat=c_pc_feat,
        )

    print(f"输入形状: x={x.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == (batch_size, num_points, 2), "输出形状错误"
    print("   ✅ 原始 Perceiver 测试通过")

    print("\n" + "=" * 60)
    print("向后兼容性测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_enhanced_components()
        test_cdm_integration()
        test_backward_compatibility()

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！增强 Perceiver 已成功集成到 CDM 中！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
