#!/bin/bash
# Generate Affordance Maps for Stage 2 Training
#
# Usage:
#   bash scripts/gen_train_contact.sh <CDM_EXP_DIR> [SAVE_DIR] [ARCH]
#
# Example:
#   bash scripts/gen_train_contact.sh outputs/2024-01-15_cdm_perceiver
#   bash scripts/gen_train_contact.sh outputs/cdm_exp data Perceiver
#   bash scripts/gen_train_contact.sh 2026-01-03_03-00-57_CDM-Perceiver-H3D-PointMamba map_pointmamba PointMamba

EXP_DIR=$1
SAVE_DIR=$2
ARCH=$3

if [ -z "$EXP_DIR" ]; then
    echo "Usage: bash scripts/gen_train_contact.sh <CDM_EXP_DIR> [SAVE_DIR] [ARCH]"
    exit 1
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="data"
fi

if [ -z "$ARCH" ]; then
    ARCH="Perceiver"
fi

SEED=2023

echo "=============================================="
echo "Generate Affordance Maps for Stage 2 Training"
echo "=============================================="
echo "CDM Exp Dir: ${EXP_DIR}"
echo "Save Dir: ${SAVE_DIR}"
echo "Architecture: ${ARCH}"
echo "=============================================="

python gen_train_contact.py hydra/job_logging=none hydra/hydra_logging=none \
    exp_dir=${EXP_DIR} \
    seed=${SEED} \
    +save_dir=${SAVE_DIR} \
    output_dir=outputs \
    diffusion.steps=500 \
    task=text_to_motion_contact_gen \
    +task.dataset.phase=train \
    model=cdm \
    model.arch=${ARCH} \
    model.scene_model.use_scene_model=False \
    model.text_model.max_length=20 \
    task.dataset.sigma=0.8

echo ""
echo "=============================================="
echo "Done! Files saved to: ${SAVE_DIR}/H3D/pred_contact/"
echo ""
echo "Now train Stage 2 with:"
echo "  task.dataset.mix_train_ratio=0.5"
echo "=============================================="
