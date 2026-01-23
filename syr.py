"""
诊断为什么这么多样本被跳过
"""
import os
import numpy as np
from collections import defaultdict

# 配置
data_dir = "data"
pred_dir = "outputs/2025-12-25_10-44-17_CDM-Perceiver-HUMANISE-step200k-PointMamba/eval/test-1230-191237/HUMANISE/pred_contact"
dataset = "HUMANISE"
phase = "test"

# 读取test.txt
txt_path = f'{data_dir}/{dataset}/{phase}.txt'
with open(txt_path, 'r') as f:
    split_ids = [int(line.strip()) for line in f.readlines()]

print(f"Total samples in {phase}.txt: {len(split_ids)}")
print("=" * 60)

# 统计
skip_reasons = defaultdict(list)
processed_samples = []

for idx in split_ids:
    # 检查GT文件
    gt_file = os.path.join(data_dir, f'{dataset}/contact_motion/contacts/{idx:0>5}.npz')
    if not os.path.exists(gt_file):
        skip_reasons['gt_missing'].append(idx)
        continue

    # 检查预测文件
    pred_file = os.path.join(pred_dir, f'{idx:0>5}.npy')
    if not os.path.exists(pred_file):
        skip_reasons['pred_missing'].append(idx)
        continue

    # 加载并检查形状
    try:
        gt_data = np.load(gt_file)
        gt_dist = gt_data['dist']

        pred_dist = np.load(pred_file)
        if pred_dist.ndim == 3:
            pred_dist = pred_dist[0]

        # 检查形状
        if pred_dist.shape[0] != gt_dist.shape[0]:
            skip_reasons['point_mismatch'].append(idx)
            continue

        # 应用first策略后的形状
        if pred_dist.shape[1] != gt_dist.shape[1]:
            gt_dist_adjusted = gt_dist[:, :pred_dist.shape[1]]
            if pred_dist.shape != gt_dist_adjusted.shape:
                skip_reasons['shape_mismatch_after_adjust'].append(idx)
                continue

        processed_samples.append(idx)

    except Exception as e:
        skip_reasons['load_error'].append((idx, str(e)))
        continue

# 打印结果
print("\n📊 Statistics:")
print("=" * 60)
print(f"✅ Processed: {len(processed_samples)}")
print(f"❌ Skipped: {len(split_ids) - len(processed_samples)}")
print()

print("📋 Skip Reasons:")
print("=" * 60)
for reason, samples in skip_reasons.items():
    print(f"{reason}: {len(samples)}")
    if len(samples) <= 10:
        print(f"  Sample IDs: {samples}")
    else:
        print(f"  First 10 IDs: {samples[:10]}")
        print(f"  Last 10 IDs: {samples[-10:]}")
    print()

# 检查预测文件总数
print("\n📁 File Count Check:")
print("=" * 60)
if os.path.exists(pred_dir):
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.npy')]
    print(f"Total .npy files in pred_dir: {len(pred_files)}")
    print(f"First 5 files: {sorted(pred_files)[:5]}")
    print(f"Last 5 files: {sorted(pred_files)[-5:]}")
else:
    print(f"❌ pred_dir does not exist: {pred_dir}")

# 检查GT文件总数
gt_dir = os.path.join(data_dir, f'{dataset}/contact_motion/contacts')
if os.path.exists(gt_dir):
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.npz')]
    print(f"\nTotal .npz files in GT dir: {len(gt_files)}")
else:
    print(f"❌ GT dir does not exist: {gt_dir}")

# 检查test.txt中的ID范围
print("\n📝 ID Range in test.txt:")
print("=" * 60)
print(f"Min ID: {min(split_ids)}")
print(f"Max ID: {max(split_ids)}")
print(f"First 10 IDs: {split_ids[:10]}")
print(f"Last 10 IDs: {split_ids[-10:]}")

# 对比：哪些ID有预测但不在test.txt中
if os.path.exists(pred_dir):
    pred_ids = set([int(f.split('.')[0]) for f in os.listdir(pred_dir) if f.endswith('.npy')])
    test_ids = set(split_ids)

    extra_pred = pred_ids - test_ids
    missing_pred = test_ids - pred_ids

    print("\n🔍 ID Matching:")
    print("=" * 60)
    print(f"IDs in pred_dir but not in test.txt: {len(extra_pred)}")
    if len(extra_pred) <= 20:
        print(f"  {sorted(extra_pred)}")

    print(f"\nIDs in test.txt but not in pred_dir: {len(missing_pred)}")
    if len(missing_pred) <= 20:
        print(f"  {sorted(missing_pred)}")
    else:
        print(f"  First 20: {sorted(missing_pred)[:20]}")