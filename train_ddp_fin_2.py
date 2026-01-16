import os
import hydra
import torch
import random
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.distributed import DistributedSampler

from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model_and_diffusion
from utils.io import mkdir_if_not_exists, Board
from utils.training import TrainLoop
from utils.misc import compute_repr_dimesion


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """ Main function

    Args:
        cfg: configuration dict
    """
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)

    ## set rank and device
    cfg.gpu = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(cfg.gpu)
    device = torch.device('cuda', cfg.gpu)
    torch.distributed.init_process_group(backend='nccl')

    ## set output logger and plot board
    if cfg.gpu == 0:
        logger.remove(handler_id=0)  # remove default handler
        mkdir_if_not_exists(cfg.log_dir)
        mkdir_if_not_exists(cfg.ckpt_dir)
        mkdir_if_not_exists(cfg.eval_dir)

        logger.add(cfg.log_dir + '/runtime.log')
        Board().create_board(cfg.platform, project=cfg.project, log_dir=cfg.log_dir)

        ## Begin training progress
        logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')
        logger.info('[Train] ==> Beign training..')

    # prepare training dataset
    train_dataset = create_dataset(cfg.task.dataset, cfg.task.train.phase, gpu=cfg.gpu)
    if cfg.gpu == 0:
        logger.info(f'Load train dataset size: {len(train_dataset)}')
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_dataloader = train_dataset.get_dataloader(
        sampler=train_sampler,
        batch_size=cfg.task.train.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.train.num_workers,
        pin_memory=True,
    )

    ## create model and optimizer
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)

    # ==============================================================================
    # [Step 1] 加载权重 (部分加载 / Partial Load)
    # ==============================================================================
    resume_ckpt = cfg.task.train.get('resume_ckpt', None)

    if resume_ckpt and os.path.exists(resume_ckpt):
        if cfg.gpu == 0:
            logger.info(f"[Transfer] Loading weights from {resume_ckpt}...")

        checkpoint = torch.load(resume_ckpt, map_location=device)

        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint

        model_dict = model.state_dict()
        pretrained_dict_filtered = {}

        # [核心逻辑修改] 定义要跳过的层
        # 目标：前2层(0-1)是Transformer (复用旧权重)，后3层(2-4)是Bi-Mamba或其他结构 (新结构/重置参数)
        # 这里的 encoder_layers.X 对应模型的层索引
        layers_to_skip = ["encoder_layers.2", "encoder_layers.3", "encoder_layers.4"]

        for k, v in pretrained_dict.items():
            k_clean = k.replace("module.", "")

            is_skip = False
            for skip_prefix in layers_to_skip:
                if skip_prefix in k_clean:
                    is_skip = True
                    break

            if is_skip:
                # if cfg.gpu == 0:
                #     logger.info(f"[Transfer] Skipping layer: {k_clean}")
                continue

            if k_clean in model_dict:
                if v.shape == model_dict[k_clean].shape:
                    pretrained_dict_filtered[k_clean] = v
                else:
                    if cfg.gpu == 0:
                        logger.warning(
                            f"[Transfer] Shape mismatch for {k_clean}: ckpt {v.shape} vs model {model_dict[k_clean].shape}")
            else:
                pass

        model.load_state_dict(pretrained_dict_filtered, strict=False)

        if cfg.gpu == 0:
            logger.info(f"[Transfer] Successfully loaded {len(pretrained_dict_filtered)} keys.")
            # 更新日志信息以匹配新的逻辑
            logger.info("[Transfer] Layers 2, 3 & 4 are initialized from scratch.")
            logger.info("[Transfer] Fine-tuning Mode: Pre-trained layers 0-1 loaded, all layers are trainable.")

    elif resume_ckpt and cfg.gpu == 0:
        logger.warning(f"[Transfer] resume_ckpt path provided but file not found: {resume_ckpt}")

    # ==============================================================================
    # [Step 2] 移除冻结逻辑 (Fine-tuning 模式)
    # ==============================================================================
    # 保持不变：确保所有参数都参与训练
    if cfg.gpu == 0:
        logger.info("-" * 40)
        logger.info("Trainable Parameters Check:")
        count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                count += param.numel()
        logger.info(f"Total Trainable Params: {count / 1e6:.2f} M")
        logger.info("-" * 40)

    # ==============================================================================
    # [Step 3] 清空 resume_ckpt，防止加载旧的优化器状态导致报错
    # ==============================================================================
    if cfg.task.train.get('resume_ckpt') is not None:
        cfg.task.train.resume_ckpt = None

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[cfg.gpu], output_device=cfg.gpu, find_unused_parameters=True, broadcast_buffers=False)

    ## start training
    TrainLoop(
        cfg=cfg.task.train,
        model=model,
        diffusion=diffusion,
        dataloader=train_dataloader,
        device=device,
        save_dir=cfg.ckpt_dir,
        gpu=cfg.gpu,
        is_distributed=True,
    ).run_loop()

    ## Training is over!
    if cfg.gpu == 0:
        Board().close()  # close board
        logger.info('[Train] ==> End training..')


if __name__ == '__main__':
    SEED = 2023
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    main()