#!/bin/bash
# Generate Affordance Maps for Stage 2 Training (Multi-GPU Support)
#
# Usage:
#   单GPU:  bash scripts/gen_train_contact.sh <CDM_EXP_DIR> [SAVE_DIR] [ARCH] [BATCH_SIZE]
#   双GPU:  bash scripts/gen_train_contact.sh <CDM_EXP_DIR> [SAVE_DIR] [ARCH] [BATCH_SIZE] multi
#
# Example:
#   bash scripts/gen_train_contact.sh outputs/2024-01-15_cdm_perceiver
#   bash scripts/gen_train_contact.sh outputs/cdm_exp data Perceiver 64
#   bash scripts/gen_train_contact.sh outputs/2026-01-03_03-00-57_CDM-Perceiver-H3D-PointMamba map_pointmamba PointMamba 64 multi

EXP_DIR=$1
SAVE_DIR=$2
ARCH=$3
BATCH_SIZE=$4
MULTI_GPU=$5

if [ -z "$EXP_DIR" ]; then
    echo "Usage: bash scripts/gen_train_contact.sh <CDM_EXP_DIR> [SAVE_DIR] [ARCH] [BATCH_SIZE] [multi]"
    exit 1
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="data"
fi

if [ -z "$ARCH" ]; then
    ARCH="Perceiver"
fi

if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=32
fi

SEED=2023

echo "=============================================="
echo "Generate Affordance Maps for Stage 2 Training"
echo "=============================================="
echo "CDM Exp Dir: ${EXP_DIR}"
echo "Save Dir: ${SAVE_DIR}"
echo "Architecture: ${ARCH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Multi-GPU: ${MULTI_GPU:-single}"
echo "=============================================="

if [ "$MULTI_GPU" = "multi" ]; then
    echo "Starting 2 GPU processes in parallel..."

    # GPU 0: 处理偶数batch (split_id=0)
    python gen_train_contact.py hydra/job_logging=none hydra/hydra_logging=none \
        exp_dir=${EXP_DIR} \
        seed=${SEED} \
        gpu=0 \
        +save_dir=${SAVE_DIR} \
        +num_splits=2 \
        +split_id=0 \
        output_dir=outputs \
        diffusion.steps=500 \
        task=text_to_motion_contact_gen \
        +task.dataset.phase=train \
        task.test.batch_size=${BATCH_SIZE} \
        model=cdm \
        model.arch=${ARCH} \
        model.scene_model.use_scene_model=False \
        model.text_model.max_length=20 \
        task.dataset.sigma=0.8 > gen_map_gpu0.log 2>&1 &

    PID0=$!
    echo "GPU 0 started (PID: $PID0), log: gen_map_gpu0.log"

    # GPU 1: 处理奇数batch (split_id=1)
    python gen_train_contact.py hydra/job_logging=none hydra/hydra_logging=none \
        exp_dir=${EXP_DIR} \
        seed=${SEED} \
        gpu=1 \
        +save_dir=${SAVE_DIR} \
        +num_splits=2 \
        +split_id=1 \
        output_dir=outputs \
        diffusion.steps=500 \
        task=text_to_motion_contact_gen \
        +task.dataset.phase=train \
        task.test.batch_size=${BATCH_SIZE} \
        model=cdm \
        model.arch=${ARCH} \
        model.scene_model.use_scene_model=False \
        model.text_model.max_length=20 \
        task.dataset.sigma=0.8 > gen_map_gpu1.log 2>&1 &

    PID1=$!
    echo "GPU 1 started (PID: $PID1), log: gen_map_gpu1.log"

    echo ""
    echo "Both GPUs running in background. Monitor with:"
    echo "  tail -f gen_map_gpu0.log"
    echo "  tail -f gen_map_gpu1.log"
    echo "  ls ${SAVE_DIR}/H3D/pred_contact/ | wc -l"
    echo ""
    echo "Waiting for both processes to finish..."
    wait $PID0 $PID1
    echo "Done!"
else
    # 单GPU模式
    python gen_train_contact.py hydra/job_logging=none hydra/hydra_logging=none \
        exp_dir=${EXP_DIR} \
        seed=${SEED} \
        +save_dir=${SAVE_DIR} \
        output_dir=outputs \
        diffusion.steps=500 \
        task=text_to_motion_contact_gen \
        +task.dataset.phase=train \
        task.test.batch_size=${BATCH_SIZE} \
        model=cdm \
        model.arch=${ARCH} \
        model.scene_model.use_scene_model=False \
        model.text_model.max_length=20 \
        task.dataset.sigma=0.8
fi

echo ""
echo "=============================================="
echo "Done! Files saved to: ${SAVE_DIR}/H3D/pred_contact/"
echo ""
echo "Now train Stage 2 with:"
echo "  task.dataset.mix_train_ratio=0.5"
echo "=============================================="
