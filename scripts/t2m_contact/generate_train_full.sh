#!/bin/bash

EXP_DIR=$1
SEED=${2:-2023}

if [ -z "$EXP_DIR" ]; then
    echo "用法: bash generate_train.sh <model_dir> [seed]"
    echo "示例: bash generate_train.sh outputs/PointMamba-ADM/ 2023"
    exit 1
fi

echo "========================================"
echo "生成训练集affordance maps"
echo "模型目录: ${EXP_DIR}"
echo "输出目录: data/H3D/pred_contact_PointMamba"
echo "预计耗时: 2-3小时"
echo "========================================"

python test.py \
    hydra/job_logging=none hydra/hydra_logging=none \
    exp_dir=${EXP_DIR} \
    seed=${SEED} \
    output_dir=data/H3D/pred_contact_PointMamba \
    diffusion.steps=500 \
    task=text_to_motion_contact_gen \
    task.dataset.ratio=1.0 \
    task.dataset.sigma=0.5 \
    task.test.batch_size=32 \
    task.evaluator.k_samples=0 \
    task.evaluator.eval_nbatch=10000 \
    task.evaluator.save_results=true \
    model=cdm \
    model.arch=PointMamba \
    model.scene_model.use_scene_model=False \
    model.text_model.max_length=20  \
    model.arch_pointmamba.last_dim=256

EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    FILE_COUNT=$(ls data/H3D/pred_contact_PointMamba/*.npy 2>/dev/null | wc -l)
    echo "✅ 生成完成！"
    echo "文件数量: ${FILE_COUNT}"
    echo ""

    if [ $FILE_COUNT -gt 20000 ]; then
        echo "接下来执行:"
        echo "  1. 验证: python verify_generation.py"
        echo "  2. 替换: mv data/H3D/pred_contact data/H3D/pred_contact_old"
        echo "         mv data/H3D/pred_contact_PointMamba data/H3D/pred_contact"
        echo "  3. 训练: bash scripts/t2m_contact_motion/train_ddp.sh CMDM-PointMamba 29500"
    else
        echo "⚠️  文件数量偏少，预期 ~24000 个"
    fi
else
    echo "❌ 生成失败，退出码: ${EXIT_CODE}"
fi
echo "========================================"