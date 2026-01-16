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
    # [新增] 自定义权重加载逻辑 (Partial Load for Mamba Transfer)
    # ==============================================================================

    # [关键修改 1]: 正确获取路径。这里去读 yaml 里的配置，而不是把路径写死在 get 的 key 里
    resume_ckpt = cfg.task.train.get('resume_ckpt', None)

    # 或者，如果你确实想在这里写死路径（不推荐，但可行），请解开下面这行的注释并注释掉上面那行：
    # resume_ckpt = '/home/supermicro/syr/git-sapce/afford-motion/outputs/2025-09-15_15-05-55_RTX4090-real/ckpt/model300000.pt'

    # 只有当 resume_ckpt 不为空且文件存在时才加载
    if resume_ckpt and os.path.exists(resume_ckpt):
        # 为了避免多卡重复打印，只让 rank 0 打印日志
        if cfg.gpu == 0:
            logger.info(f"[Transfer] Loading weights from {resume_ckpt}...")

        # 加载 checkpoint 到当前 GPU
        checkpoint = torch.load(resume_ckpt, map_location=device)

        # 1. 提取权重字典
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint

        # 2. 获取当前新模型的字典
        model_dict = model.state_dict()
        pretrained_dict_filtered = {}

        # 3. 定义要跳过的层 (旧模型的第5层 Transformer，即索引 4)
        layers_to_skip = ["encoder_layers.4"]

        for k, v in pretrained_dict.items():
            # DDP 保存的权重通常带有 'module.' 前缀，但当前 model 还没包 DDP，所以要去掉
            k_clean = k.replace("module.", "")

            # 检查是否属于要跳过的层
            is_skip = False
            for skip_prefix in layers_to_skip:
                if skip_prefix in k_clean:
                    is_skip = True
                    break

            if is_skip:
                if cfg.gpu == 0:
                    logger.info(f"[Transfer] Skipping layer: {k_clean}")
                continue

            # 4. 匹配并加载
            if k_clean in model_dict:
                if v.shape == model_dict[k_clean].shape:
                    pretrained_dict_filtered[k_clean] = v
                else:
                    if cfg.gpu == 0:
                        logger.warning(
                            f"[Transfer] Shape mismatch for {k_clean}: ckpt {v.shape} vs model {model_dict[k_clean].shape}")
            else:
                pass  # 忽略新模型中不存在的 key

        # 5. 加载权重 (strict=False)
        model.load_state_dict(pretrained_dict_filtered, strict=False)

        if cfg.gpu == 0:
            logger.info(f"[Transfer] Successfully loaded {len(pretrained_dict_filtered)} keys.")
            logger.info("[Transfer] Layer 4 (Mamba) is initialized from scratch.")

    elif resume_ckpt and cfg.gpu == 0:
        logger.warning(f"[Transfer] resume_ckpt path provided but file not found: {resume_ckpt}")

    # ==============================================================================
    # [关键修改 2] 清空 resume_ckpt，防止 TrainLoop 加载旧的优化器
    # ==============================================================================
    # 这一步非常重要！必须把配置里的 resume_ckpt 抹掉
    if cfg.task.train.get('resume_ckpt') is not None:
        cfg.task.train.resume_ckpt = None
    # ==============================================================================
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