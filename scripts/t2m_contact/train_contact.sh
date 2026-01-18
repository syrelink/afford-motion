#!/bin/bash

# 训练脚本：生成 CDM 第一阶段训练数据集
# 使用 ContactPointMamba 架构
# 支持多GPU并行生成

EXP_DIR=$1
GPU=$2
SEED=$3

# 设置默认值
if [ -z "$GPU" ]
then
    GPU=0
fi

if [ -z "$SEED" ]
then
    SEED=2023
fi

echo "=========================================="
echo "CDM 第一阶段训练数据生成 (多GPU)"
echo "=========================================="
echo "Checkpoint 目录: ${EXP_DIR}"
echo "GPU: ${GPU}"
echo "随机种子: ${SEED}"
echo "=========================================="
echo ""

# 检查 checkpoint 目录是否存在
if [ ! -d "${EXP_DIR}/ckpt" ]; then
    echo "错误: 找不到 checkpoint 目录 ${EXP_DIR}/ckpt"
    echo "请确保 checkpoint 文件存在于该目录中"
    exit 1
fi

# 检查是否有 checkpoint 文件
if [ ! -f "${EXP_DIR}/ckpt/model300000.pt" ]; then
    echo "警告: 在 ${EXP_DIR}/ckpt/ 中未找到 model300000.pt 文件"
    echo "请确保 checkpoint 文件存在"
    echo "继续执行..."
fi

# 运行训练数据生成
# 处理GPU参数：如果是多个GPU，转换为列表格式
if [[ "${GPU}" == *","* ]]; then
    GPU_PARAM="[${GPU//,/, }]"
else
    GPU_PARAM="${GPU}"
fi

python train_contact_gen.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir=${EXP_DIR} \
            seed=${SEED} \
            gpu=${GPU_PARAM} \
            output_dir=outputs \
            exp_name=cdm_first_stage \
            task=contact_gen_train \
            task.train.batch_size=64 \
            task.train.num_workers=8 \
            task.train.phase=train \
            model=cdm \
            model.arch=PointMamba \
            model.scene_model.use_scene_model=False \
            model.text_model.max_length=20 \
            task.dataset.sigma=0.8 \
            task.dataset.num_points=8192 \
            task.dataset.use_color=true \
            task.dataset.use_openscene=false \
            diffusion.steps=1000 \
            diffusion.noise_schedule=cosine

echo ""
echo "=========================================="
echo "训练数据生成完成!"
echo "=========================================="
