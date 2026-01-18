#!/usr/bin/env python3
"""
专门用于生成训练集接触点数据的脚本
基于test.py修改，但专门为训练集生成优化
"""

import os
import sys
import glob
import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from natsort import natsorted

from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model_and_diffusion
from utils.io import mkdir_if_not_exists, time_str
from utils.evaluate import create_evaluator
from utils.training import load_ckpt
from utils.misc import compute_repr_dimesion


def generate_train_contacts(cfg: DictConfig) -> None:
    """ 生成训练集接触点数据 """

    # 创建专门的输出目录
    save_dir = os.path.join(cfg.output_dir, 'train_contacts-' + time_str(Y=False))
    mkdir_if_not_exists(save_dir)
    logger.add(os.path.join(save_dir, 'generate.log'))
    logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')
    logger.info('[Generate] ==> Begin generating training contacts..')

    # 检查是否已有部分生成的数据（断点续传）
    checkpoint_file = os.path.join(save_dir, 'checkpoint.txt')
    start_batch = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_batch = int(f.read().strip())
        logger.info(f'Found checkpoint, resuming from batch {start_batch}')
    else:
        logger.info(f'No checkpoint found, starting from batch 0')

    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'

    # 关键：使用'train' phase
    train_dataset = create_dataset(cfg.task.dataset, 'train', gpu=cfg.gpu, **cfg.task.test)
    logger.info(f'Load train dataset size: {len(train_dataset)}')

    train_dataloader = train_dataset.get_dataloader(
        batch_size=cfg.task.test.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.test.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    ## 创建模型并加载checkpoint
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)

    ckpts = natsorted(glob.glob(os.path.join(cfg.exp_dir, 'ckpt', 'model*.pt')))
    assert len(ckpts) > 0, 'No checkpoint found.'
    load_ckpt(model, ckpts[-1])
    logger.info(f'Load checkpoint from {ckpts[-1]}')

    ## 创建评估器
    evaluator = create_evaluator(cfg.task, device=device)

    ## 生成样本
    model.eval()
    sample_fn = diffusion.p_sample_loop

    B = train_dataloader.batch_size
    sample_list = []
    k_samples_list = []
    if evaluator.k_samples > 0:
        k_samples_idxs = list(range(
            evaluator.num_k_samples // B))
    else:
        k_samples_idxs = []
    logger.info(f'k_samples_idxs: {k_samples_idxs}')

    for i, data in enumerate(train_dataloader):
        # 断点续传：跳过已处理的批次
        if i < start_batch:
            if i % 10 == 0:
                logger.info(f"Skipping batch {i} (already processed)")
            continue

        logger.info(f"batch index: {i}, is k_sample_batch: {i in k_samples_idxs}, case index: {data['info_index']}")
        x = data['x']

        x_kwargs = {}
        if 'x_mask' in data:
            x_kwargs['x_mask'] = data['x_mask'].to(device)

        for key in data:
            if key.startswith('c_') or key.startswith('info_'):
                if torch.is_tensor(data[key]):
                    x_kwargs[key] = data[key].to(device)
                else:
                    x_kwargs[key] = data[key]

        use_k_sample = i in k_samples_idxs
        repeat_times = evaluator.k_samples if use_k_sample else 1

        sample_list_np = []
        k_samples_list_np = []
        for k in range(repeat_times):
            if cfg.model.name.startswith('CMDM'):
                # 修复：检查 c_pc_contact 的维度
                if len(data['c_pc_contact'].shape) == 4:
                    # 有 k 维度: [B, k, N, D]
                    x_kwargs['c_pc_contact'] = data['c_pc_contact'][:, k, :, :].to(device)
                else:
                    # 无 k 维度: [B, N, D]
                    x_kwargs['c_pc_contact'] = data['c_pc_contact'].to(device)

            sample = sample_fn(
                model,
                x.shape,
                clip_denoised=False,
                noise=None,
                model_kwargs=x_kwargs,
                progress=True,
            )

            if k == 0:
                for bsi in range(B):
                    # 修复：检查 sample 的维度
                    if bsi < sample.shape[0]:
                        sample_list_np.append(sample[bsi].cpu().numpy())
                    else:
                        logger.warning(f"Warning: bsi {bsi} >= sample.shape[0] {sample.shape[0]}")

            if use_k_sample:
                for bsi in range(B):
                    if bsi < sample.shape[0]:
                        k_samples_list_np.append(sample[bsi].cpu().numpy())
                    else:
                        logger.warning(f"Warning (k_samples): bsi {bsi} >= sample.shape[0] {sample.shape[0]}")

        ## 1 sample
        # 修复：使用实际的样本数量，而不是 batch_size
        actual_batch_size = len(sample_list_np)
        for bsi in range(actual_batch_size):
            res_dict = {'sample': sample_list_np[bsi]}
            for key in data:
                if torch.is_tensor(data[key]):
                    res_dict[key] = data[key][bsi].cpu().numpy()
                else:
                    res_dict[key] = data[key][bsi]
            sample_list.append(res_dict)

        ## k samples
        if use_k_sample:
            actual_batch_size = len(sample_list_np)
            for bsi in range(actual_batch_size):
                res_dict = {'k_samples': np.stack(k_samples_list_np[bsi::actual_batch_size])}
                for key in data:
                    if torch.is_tensor(data[key]):
                        res_dict[key] = data[key][bsi].cpu().numpy()
                    else:
                        res_dict[key] = data[key][bsi]
                k_samples_list.append(res_dict)

        ## 保存断点（每10个batch保存一次）
        if (i + 1) % 10 == 0:
            with open(checkpoint_file, 'w') as f:
                f.write(str(i + 1))
            logger.info(f"Checkpoint saved: batch {i + 1}")

        ## 如果达到最大批次则停止
        if i + 1 >= evaluator.eval_nbatch:
            break

    ## 保存结果
    evaluator.evaluate(sample_list, k_samples_list, save_dir, train_dataloader, device=device)
    logger.info(f"✅ 生成完成！数据保存在: {save_dir}")

    # 统计生成的文件数量
    pred_contact_dir = os.path.join(save_dir, 'H3D/pred_contact')
    if os.path.exists(pred_contact_dir):
        file_count = len([f for f in os.listdir(pred_contact_dir) if f.endswith('.npy')])
        logger.info(f"生成文件数量: {file_count}")

    return save_dir


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """ 主函数 """
    ## 设置随机种子
    SEED = cfg.seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    ## 计算建模维度
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)

    ## 设置输出目录
    mkdir_if_not_exists(cfg.output_dir)

    save_dir = generate_train_contacts(cfg)

    # 提供使用说明
    print("\n" + "="*60)
    print("✅ 训练集接触点数据生成完成！")
    print(f"数据目录: {save_dir}")
    print("\n下一步操作:")
    print(f"1. 链接到训练目录: ln -sf $(pwd)/{save_dir}/H3D/pred_contact data/H3D/pred_contact")
    print("2. 配置 stage2 训练使用混合数据 (mix_train_ratio: 0.5)")
    print("3. 训练 stage2 模型")
    print("="*60)


if __name__ == '__main__':
    import torch
    import random
    import numpy as np

    main()