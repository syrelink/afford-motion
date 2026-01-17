#!/usr/bin/env python3
"""
使用增强 Perceiver 训练 Stage 1 (CDM) 的示例脚本

用法:
    python scripts/train_enhanced_perceiver.py
    python scripts/train_enhanced_perceiver.py --config-name=contact_gen_enhanced
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cdm import CDM
from datasets.base import BaseDataset
from utils.training import train_one_epoch, validate
from utils.evaluate import evaluate_fid


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    """主训练函数"""

    print("=" * 60)
    print("训练增强 Perceiver (Stage 1 - CDM)")
    print("=" * 60)

    # 打印配置
    print("\n配置:")
    print(OmegaConf.to_yaml(cfg))

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 创建模型
    print("\n创建模型...")
    model = CDM(cfg.model, device=device).to(device)

    # 打印模型信息
    print(f"\n模型架构: {cfg.model.arch}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 数据加载
    print("\n加载数据...")
    train_dataset = BaseDataset(cfg.data, split='train')
    val_dataset = BaseDataset(cfg.data, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_epochs,
    )

    # 损失函数
    criterion = nn.MSELoss()

    # 训练循环
    best_fid = float('inf')
    best_epoch = 0

    print("\n开始训练...")
    print("=" * 60)

    for epoch in range(cfg.training.num_epochs):
        # 训练
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            cfg=cfg,
        )

        # 验证
        val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # 计算 FID（每 5 个 epoch）
        if (epoch + 1) % 5 == 0 or epoch == cfg.training.num_epochs - 1:
            print(f"\n计算 FID (Epoch {epoch + 1})...")
            fid_score = evaluate_fid(
                model=model,
                dataloader=val_loader,
                device=device,
                cfg=cfg,
            )
            print(f"FID: {fid_score:.4f}")

            # 保存最佳模型
            if fid_score < best_fid:
                best_fid = fid_score
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fid': fid_score,
                    'loss': val_loss,
                }, 'best_model.pth')
                print(f"保存最佳模型 (FID: {best_fid:.4f})")

        # 更新学习率
        scheduler.step()

        # 打印进度
        print(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")

    print("\n" + "=" * 60)
    print(f"训练完成!")
    print(f"最佳 FID: {best_fid:.4f} (Epoch {best_epoch})")
    print("=" * 60)


if __name__ == "__main__":
    main()
