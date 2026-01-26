#!/bin/bash
# train_bimamba.sh - BiMamba训练脚本

EXP_NAME=$1
PORT=$2
CHECKPOINT=$3

if [ -z "$PORT" ]
then
    PORT=29500
fi

# 打印配置
echo "=========================================="
echo "BiMamba训练配置"
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

    # +task.dataset.pred_contact_dir=map_pointmamba \
    # task.dataset.mix_train_ratio=0.5 \

# 构建命令（使用bimamba架构）
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
    model=cmdm \
    model.arch='bimamba' \
    model.data_repr='h3d' \
    model.text_model.max_length=20 \
    model.latent_dim=512 \
    model.num_layers=[3,2] \
    model.dim_feedforward=1024 \
    model.dropout=0.15 \
    model.mamba_layers=2 \
    model.mamba_d_state=16 \
    model.mamba_d_conv=4 \
    model.mamba_expand=2 \
    model.mamba_drop_path=0.05 \


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
