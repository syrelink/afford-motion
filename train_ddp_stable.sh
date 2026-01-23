#!/bin/bash
# train_ddp_stable.sh - 稳定训练脚本（基于官方脚本修改）

EXP_NAME=$1
PORT=$2
CHECKPOINT=$3

if [ -z "$PORT" ]
then
    PORT=29500
fi

# 打印配置
echo "=========================================="
echo "稳定训练配置"
echo "=========================================="
echo "实验名称: $EXP_NAME"
echo "端口: $PORT"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
else
    echo "Checkpoint: 从头训练"
fi
echo "=========================================="
echo ""

# 构建命令（基于官方脚本，最小改动）
CMD="CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} \
    train_ddp.py \
    hydra/job_logging=none hydra/hydra_logging=none \
    exp_name=${EXP_NAME} \
    output_dir=outputs \
    platform=TensorBoard \
    diffusion.steps=1000 \
    task=text_to_motion_contact_motion_gen \
    task.dataset.sigma=0.8 \
    task.train.batch_size=32 \
    task.train.max_steps=600000 \
    task.train.save_every_step=25000 \
    task.dataset.train_transforms=['RandomEraseLang','RandomEraseContact','NumpyToTensor'] \
    +task.dataset.pred_contact_dir=map_pointmamba \
    task.dataset.mix_train_ratio=0.5 \
    model=cmdm \
    model.arch='dit' \
    model.data_repr='h3d' \
    model.text_model.max_length=20 \
    model.latent_dim=512 \
    model.num_layers=[1,1,1,1,1] \
    model.dim_feedforward=1024 \
    model.dropout=0.15 \
    model.dit_drop_path=0.05 \
    model.dit_use_cross_attn_pooling=true \
    model.condition_embedder.use_cross_attn_pooling=true \
    model.condition_embedder.num_latents=64 \
    model.condition_embedder.fusion_method='cross_attn' \
    training.lr=3e-5 \
    training.grad_clip=1.0 \
    training.warmup_steps=2000 \
    training.weight_decay=1e-4 \
    training.lr_scheduler='cosine' \
    training.early_stopping.enabled=true \
    training.early_stopping.patience=5 \
    training.early_stopping.min_delta=0.01 \
    training.eval_every_epochs=20 \
    training.eval_num_samples=500 \
    training.save_best=true \
    training.best_metric='fid'"

# 添加checkpoint
if [[ -n "$CHECKPOINT" ]]; then
    CMD="$CMD checkpoint.path=$CHECKPOINT"
fi

# 执行命令
echo "开始训练..."
echo "日志: outputs/${EXP_NAME}/logs/runtime.log"
echo "TensorBoard: tensorboard --logdir=outputs/${EXP_NAME}/logs --port=6006"
echo ""

# 运行训练
$CMD
