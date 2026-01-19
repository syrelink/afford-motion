"""
Generate Affordance Maps for Stage 2 Training

Usage:
    python gen_train_contact.py \
        exp_dir=outputs/your_cdm_exp \
        task.dataset.phase=train \
        save_dir=data
"""

import os, glob, hydra
import torch
import random
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from natsort import natsorted

from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model_and_diffusion
from utils.io import mkdir_if_not_exists, time_str
from utils.training import load_ckpt
from utils.misc import compute_repr_dimesion


def train_gen_map(cfg: DictConfig) -> None:
    """Generate affordance maps for training set (Stage 2 training data)

    This function generates predicted contact/affordance maps from a trained
    Stage 1 (CDM) model for the training set. Output files are saved incrementally
    to avoid memory issues.

    Output directory structure:
        {save_dir}/H3D/pred_contact/{name}-{caption_index}.npy

    Output shape: (1, num_points, contact_dim)

    Args:
        cfg: configuration dict
    """
    # Setup save directory
    save_dir = cfg.get('save_dir', 'data')  # 默认保存到 data 目录
    contact_save_dir = os.path.join(save_dir, 'H3D/pred_contact')
    os.makedirs(contact_save_dir, exist_ok=True)  # 递归创建目录

    logger.info(f'[Generate] Save directory: {contact_save_dir}')
    logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')

    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'

    # Prepare training dataset (使用train phase)
    phase = cfg.task.dataset.get('phase', 'train')
    train_dataset = create_dataset(cfg.task.dataset, phase, gpu=cfg.gpu)
    logger.info(f'Load {phase} dataset size: {len(train_dataset)}')

    train_dataloader = train_dataset.get_dataloader(
        batch_size=cfg.task.test.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.test.num_workers,
        pin_memory=True,
        shuffle=False,  # 不打乱，保证可复现
    )

    # Create model and load checkpoint
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)

    ckpts = natsorted(glob.glob(os.path.join(cfg.exp_dir, 'ckpt', 'model*.pt')))
    assert len(ckpts) > 0, 'No checkpoint found.'
    load_ckpt(model, ckpts[-1])
    logger.info(f'Load checkpoint from {ckpts[-1]}')

    # Sample
    model.eval()
    sample_fn = diffusion.p_sample_loop

    B = train_dataloader.batch_size
    generated_count = 0
    skipped_count = 0

    for i, data in enumerate(train_dataloader):
        logger.info(f"[{i+1}/{len(train_dataloader)}] batch index: {i}, case index: {data['info_index']}")

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

        # Generate one sample per input
        sample = sample_fn(
            model,
            x.shape,
            clip_denoised=False,
            noise=None,
            model_kwargs=x_kwargs,
            progress=True,
        )

        # 边生成边保存 (save incrementally)
        for bsi in range(min(B, sample.shape[0])):
            name = data['info_index'][bsi]
            caption_index = data['info_caption_index'][bsi]
            if isinstance(caption_index, torch.Tensor):
                caption_index = caption_index.item()

            save_path = os.path.join(contact_save_dir, f'{name}-{caption_index}.npy')

            # Skip if already exists
            if os.path.exists(save_path):
                skipped_count += 1
                continue

            # Get sample and denormalize (复用evaluate.py的逻辑)
            contact = sample[bsi].cpu().numpy()
            contact = train_dataloader.dataset.denormalize(contact, clip=True)

            # Convert to distance format
            if train_dataloader.dataset.use_raw_dist:
                dist = contact.copy()
            else:
                # Reverse Gaussian: contact = exp(-0.5 * dist^2 / sigma^2)
                # => dist = sqrt(-2 * log(contact) * sigma^2)
                dist = np.sqrt(-2 * np.log(np.clip(contact, 1e-6, 1.0)) * train_dataloader.dataset.sigma ** 2)

            # Save with shape (1, num_points, contact_dim)
            np.save(save_path, dist[None, ...].astype(np.float32))
            generated_count += 1

        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(train_dataloader)} batches, Generated: {generated_count}, Skipped: {skipped_count}")

    logger.info(f'\n[Done] Generated: {generated_count}, Skipped: {skipped_count}')
    logger.info(f'[Done] Output directory: {contact_save_dir}')


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main function"""
    # Setup random seed
    SEED = cfg.seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Compute modeling dimension
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)

    # Set output directories
    mkdir_if_not_exists(cfg.log_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)
    mkdir_if_not_exists(cfg.eval_dir)

    train_gen_map(cfg)


if __name__ == '__main__':
    main()
