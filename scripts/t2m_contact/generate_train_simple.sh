#!/bin/bash

# ============================================
# 简单的生成训练集脚本
# 用法: bash generate_train_simple.sh <模型目录> [随机种子]
# ============================================

EXP_DIR=$1
SEED=${2:-2023}

if [ -z "$EXP_DIR" ]; then
    echo "用法: bash generate_train_simple.sh <模型目录> [随机种子]"
    echo "示例: bash generate_train_simple.sh outputs/CDM-PointMamba-H3D/ 2023"
    exit 1
fi

echo "开始生成训练集接触点数据..."

# 使用专门的生成脚本
python scripts/generate_train_contacts.py \
    hydra/job_logging=none \
    hydra/hydra_logging=none \
    exp_dir=${EXP_DIR} \
    seed=${SEED} \
    output_dir=data/H3D/pred_contact_pointmamba \
    diffusion.steps=500 \
    task=text_to_motion_contact_gen \
    model=cdm \
    model.arch=PointMamba \
    model.scene_model.use_scene_model=False \
    model.text_model.max_length=20 \
    model.arch_pointmamba.last_dim=256 \
    task.dataset.sigma=0.8 \
    task.evaluator.k_samples=0 \
    task.evaluator.eval_nbatch=10000 \
    task.evaluator.num_k_samples=128 \
    task.evaluator.save_results=true

echo "生成完成！"