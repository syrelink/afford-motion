#!/bin/bash

  # ============================================
  # 生成训练集接触点数据脚本
  # 用法: bash generate_train_contacts.sh <模型目录> [随机种子]
  # 示例: bash generate_train_contacts.sh outputs/CDM-PointMamba-H3D/ 2023
  # ============================================

  EXP_DIR=$1
  SEED=${2:-2023}

  if [ -z "$EXP_DIR" ]; then
      echo "错误: 必须指定模型目录"
      echo "用法: bash generate_train_contacts.sh <模型目录> [随机种子]"
      echo "示例: bash generate_train_contacts.sh outputs/CDM-PointMamba-H3D/ 2023"
      exit 1
  fi

  if [ ! -d "$EXP_DIR" ]; then
      echo "错误: 模型目录不存在: $EXP_DIR"
      exit 1
  fi

  echo "========================================"
  echo "开始生成训练集接触点数据"
  echo "模型目录: $EXP_DIR"
  echo "随机种子: $SEED"
  echo "========================================"

  # 关键修改：
  # 1. 使用 override 参数强制修改配置
  # 2. 将数据集phase改为'train'
  # 3. 增加eval_nbatch处理全部训练数据
  # 4. 确保save_results=true
  # 5. 输出到数据目录而不是评估目录

  python test.py hydra/job_logging=none hydra/hydra_logging=none \
              exp_dir=${EXP_DIR} \
              seed=${SEED} \
              output_dir=data/H3D/pred_contact_pointmamba \
              diffusion.steps=500 \
              task=text_to_motion_contact_gen \
              model=cdm \
              model.arch=PointMamba \
              model.scene_model.use_scene_model=False \
              model.text_model.max_length=20 \
              task.dataset.sigma=0.8 \
              +task.dataset.phase='train' \                # 关键：覆盖为训练集
            #   task.evaluator.k_samples=0 \                # 不需要多样本
            #   task.evaluator.eval_nbatch=10000 \          # 关键：处理大量数据
              task.evaluator.num_k_samples=128 \
            #   +task.evaluator.save_results=true           # 关键：必须保存结果

  EXIT_CODE=$?

  echo "========================================"
  if [ $EXIT_CODE -eq 0 ]; then
      echo "✅ 生成完成！"
      echo ""
      echo "生成的数据存放在: data/H3D/pred_contact_pointmamba/"
      echo ""
      echo "目录结构:"
      echo "data/H3D/pred_contact_pointmamba/"
      echo "├── train_contacts-XXXX-XXXXXX/  # 时间戳目录"
      echo "│   ├── test.log"
      echo "│   └── H3D/pred_contact/        # 实际数据文件"
      echo "│       ├── 00001-0.npy"
      echo "│       ├── 00001-1.npy"
      echo "│       └── ..."
      echo ""
      echo "下一步操作:"
      echo "1. 验证生成的数据:"
      echo "   ls data/H3D/pred_contact_pointmamba/*/H3D/pred_contact/ | head -10"
      echo "2. 将数据链接到训练目录:"
      echo "   ln -sf \$(pwd)/data/H3D/pred_contact_pointmamba/*/H3D/pred_contact data/H3D/pred_contact"
      echo "3. 训练stage2模型:"
      echo "   bash scripts/t2m_contact_motion/train_ddp.sh CMDM-PointMamba-H3D 29500"
  else
      echo "❌ 生成失败，退出码: ${EXIT_CODE}"
      echo "请检查错误信息"
  fi
  echo "========================================"