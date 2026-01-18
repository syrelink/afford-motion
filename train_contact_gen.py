import os, glob, hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from natsort import natsorted
import torch
import random
import numpy as np

from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model_and_diffusion
from utils.io import mkdir_if_not_exists, time_str
from utils.evaluate import create_evaluator
from utils.training import load_ckpt
from utils.misc import compute_repr_dimesion


def train_contact_gen(cfg: DictConfig) -> None:
    """ Generate CDM first stage training dataset using ContactPointMamba

    Args:
        cfg: configuration dict
    """
    # Create output directory with timestamp
    train_dir = os.path.join(cfg.eval_dir, 'train-' + time_str(Y=False))
    mkdir_if_not_exists(train_dir)

    # Add logger
    logger.add(os.path.join(train_dir, 'train.log'))
    logger.info('[Configuration]\\n' + OmegaConf.to_yaml(cfg) + '\\n')
    logger.info('[Train Contact Gen] ==> Begin generating training data..')

    # Device setup
    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'

    # Prepare training dataset
    phase = cfg.task.dataset.get('phase', 'train')
    train_dataset = create_dataset(cfg.task.dataset, phase, gpu=cfg.gpu, **cfg.task.train)
    logger.info(f'Load {phase} dataset size: {len(train_dataset)}')

    train_dataloader = train_dataset.get_dataloader(
        batch_size=cfg.task.train.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.train.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    ## Create model and load checkpoint
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)

    # Load checkpoint
    ckpts = natsorted(glob.glob(os.path.join(cfg.exp_dir, 'ckpt', 'model*.pt')))
    assert len(ckpts) > 0, 'No checkpoint found.'
    load_ckpt(model, ckpts[-1])
    logger.info(f'Load checkpoint from {ckpts[-1]}')

    # Display checkpoint and output path info
    logger.info(f'\\n=== Checkpoint Info ===')
    logger.info(f'Checkpoint: {ckpts[-1]}')
    logger.info(f'Output directory: {train_dir}')
    logger.info(f'Dataset size: {len(train_dataset)}')
    logger.info(f'Batch size: {train_dataloader.batch_size}')

    ## Create evaluator
    evaluator = create_evaluator(cfg.task, device=device)

    ## Sample
    model.eval()
    sample_fn = diffusion.p_sample_loop

    B = train_dataloader.batch_size
    sample_list = []
    k_samples_list = []
    if evaluator.k_samples > 0:
        k_samples_idxs = list(range(evaluator.num_k_samples // B))
    else:
        k_samples_idxs = []
    logger.info(f'k_samples_idxs: {k_samples_idxs}')

    # Save checkpoint info to file
    checkpoint_info_path = os.path.join(train_dir, 'checkpoint_info.txt')
    with open(checkpoint_info_path, 'w') as f:
        f.write(f'Checkpoint: {ckpts[-1]}\\n')
        f.write(f'Output directory: {train_dir}\\n')
        f.write(f'Dataset size: {len(train_dataset)}\\n')
        f.write(f'Batch size: {B}\\n')
        f.write(f'\\nConfiguration:\\n{OmegaConf.to_yaml(cfg)}\\n')

    # Process each batch
    for i, data in enumerate(train_dataloader):
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
                x_kwargs['c_pc_contact'] = data['c_pc_contact'][:, k, :, :].to(device)

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
                    sample_list_np.append(sample[bsi].cpu().numpy())

            if use_k_sample:
                for bsi in range(B):
                    k_samples_list_np.append(sample[bsi].cpu().numpy())

        ## 1 sample
        for bsi in range(B):
            res_dict = {'sample': sample_list_np[bsi]}
            for key in data:
                if torch.is_tensor(data[key]):
                    res_dict[key] = data[key][bsi].cpu().numpy()
                else:
                    res_dict[key] = data[key][bsi]
            sample_list.append(res_dict)

        ## k samples
        if use_k_sample:
            for bsi in range(B):
                res_dict = {'k_samples': np.stack(k_samples_list_np[bsi::B])}
                for key in data:
                    if torch.is_tensor(data[key]):
                        res_dict[key] = data[key][bsi].cpu().numpy()
                    else:
                        res_dict[key] = data[key][bsi]
                k_samples_list.append(res_dict)

        ## Auto-save after each batch to prevent data loss
        if (i + 1) % 10 == 0:
            temp_save_path = os.path.join(train_dir, f'temp_batch_{i+1:06d}.npz')
            np.savez(temp_save_path,
                     sample_list=sample_list,
                     k_samples_list=k_samples_list,
                     batch_idx=i+1)
            logger.info(f'Auto-saved progress to {temp_save_path}')

    ## Final save of all generated data
    final_save_path = os.path.join(train_dir, 'generated_contact_maps.npz')
    np.savez(final_save_path,
             sample_list=sample_list,
             k_samples_list=k_samples_list,
             total_batches=i+1,
             dataset_size=len(train_dataset))
    logger.info(f'Final data saved to {final_save_path}')

    ## Clean up temporary files
    temp_files = glob.glob(os.path.join(train_dir, 'temp_batch_*.npz'))
    for temp_file in temp_files:
        os.remove(temp_file)
        logger.info(f'Removed temporary file: {temp_file}')

    logger.info(f'\\n=== Summary ===')
    logger.info(f'Total {len(sample_list)} samples generated')
    logger.info(f'Output directory: {train_dir}')
    logger.info(f'Final data file: {final_save_path}')


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """ Main function """
    ## setup random seed
    SEED = cfg.seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    ## compute modeling dimension
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)

    ## set output logger
    mkdir_if_not_exists(cfg.log_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)
    mkdir_if_not_exists(cfg.eval_dir)

    train_contact_gen(cfg)


if __name__ == '__main__':
    main()
