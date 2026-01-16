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


def print_model_architecture(model):
    """
    可视化打印模型每一层的类型 (Transformer vs Mamba)
    """
    logger.info("\n" + "=" * 60)
    logger.info("       [Model Architecture Layout Check]       ")
    logger.info("=" * 60)

    # 处理 DDP 包裹的情况 (虽然通常我们在 DDP 之前调用)
    real_model = model.module if hasattr(model, 'module') else model

    # 检查是否存在 encoder_layers
    if hasattr(real_model, 'encoder_layers'):
        layers = real_model.encoder_layers

        for i, layer in enumerate(layers):
            layer_class = type(layer).__name__

            # 根据类名定义标签
            if "Mamba" in layer_class:
                tag = "🐍 [MAMBA]"
                desc = f"Layer {i}: {layer_class} (Param Reset/Finetune)"
            elif "Transformer" in layer_class:
                tag = "🤖 [TRANS]"
                desc = f"Layer {i}: {layer_class} (Pre-trained)"
            else:
                tag = "❓ [OTHER]"
                desc = f"Layer {i}: {layer_class}"

            logger.info(f"{tag:<10} | {desc}")

    else:
        logger.warning("Model does not have 'encoder_layers' attribute.")

    logger.info("=" * 60 + "\n")

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    # ... (前面的初始化代码保持不变) ...
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)
    cfg.gpu = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(cfg.gpu)
    device = torch.device('cuda', cfg.gpu)
    torch.distributed.init_process_group(backend='nccl')

    if cfg.gpu == 0:
        logger.remove(handler_id=0)
        mkdir_if_not_exists(cfg.log_dir)
        mkdir_if_not_exists(cfg.ckpt_dir)
        mkdir_if_not_exists(cfg.eval_dir)
        logger.add(cfg.log_dir + '/runtime.log')
        Board().create_board(cfg.platform, project=cfg.project, log_dir=cfg.log_dir)
        logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')

    # ... (Dataset 和 Dataloader 部分保持不变) ...
    train_dataset = create_dataset(cfg.task.dataset, cfg.task.train.phase, gpu=cfg.gpu)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = train_dataset.get_dataloader(
        sampler=train_sampler,
        batch_size=cfg.task.train.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.train.num_workers,
        pin_memory=True,
    )

    ## create model
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)

    # ================= [新增] 打印架构信息 =================
    if cfg.gpu == 0:
        print_model_architecture(model)
    # =======================================================

    # ==============================================================================
    # [关键修改] 手动处理权重加载 (Transfer Learning)
    # ==============================================================================
    resume_path = cfg.task.train.get('resume_ckpt', None)

    # 只有当路径存在时才执行自定义加载
    if resume_path and os.path.exists(resume_path):
        if cfg.gpu == 0:
            logger.info(f"[Transfer] Found checkpoint: {resume_path}")
            logger.info(f"[Transfer] Starting partial weight loading...")

        # 1. 加载 Checkpoint
        checkpoint = torch.load(resume_path, map_location=device)
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint

        model_dict = model.state_dict()
        new_state_dict = {}

        # 2. 定义要保留和跳过的层
        # 你的需求：保留 Layer 0, 1 (Transformer)；跳过 Layer 2, 3, 4 (Mamba)
        # 注意：这里假设你的模型层命名是 'encoder_layers.0', 'encoder_layers.1' 等
        layers_to_skip = ["encoder_layers.2", "encoder_layers.3", "encoder_layers.4"]

        for k, v in pretrained_dict.items():
            k_clean = k.replace("module.", "")  # 去掉 DDP 前缀

            # 检查是否在跳过列表中
            is_skipped = False
            for skip_str in layers_to_skip:
                if skip_str in k_clean:
                    is_skipped = True
                    break

            if is_skipped:
                continue  # 跳过该层权重

            # 检查是否在新模型中存在且形状一致
            if k_clean in model_dict:
                if v.shape == model_dict[k_clean].shape:
                    new_state_dict[k_clean] = v
                else:
                    if cfg.gpu == 0:
                        logger.warning(f"[Transfer] Shape mismatch ignored: {k_clean}")

        # 3. 加载权重 (strict=False)
        model.load_state_dict(new_state_dict, strict=False)

        if cfg.gpu == 0:
            logger.info(f"[Transfer] Loaded {len(new_state_dict)} keys.")
            logger.info(f"[Transfer] Layers 0 & 1 loaded from Transformer.")
            logger.info(f"[Transfer] Layers 2, 3, 4 (Mamba) initialized from scratch.")

        # 4. [最重要的一步] 清空 resume_ckpt
        # 这会告诉 TrainLoop：不要加载优化器，初始化一个新的！
        cfg.task.train.resume_ckpt = None
        if cfg.gpu == 0:
            logger.info("[Transfer] cfg.resume_ckpt set to None. Optimizer will be reset.")

    # ... (后续代码保持不变) ...
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[cfg.gpu], output_device=cfg.gpu, find_unused_parameters=True, broadcast_buffers=False)

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

    if cfg.gpu == 0:
        Board().close()
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