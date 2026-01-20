"""
评估 GT affordance map 的统计指标

Usage:
    python scripts/eval_gt_contact.py --data_dir data --dataset HUMANISE --num_samples 100
"""

import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict


def evaluate_gt_contact(data_dir, dataset, num_samples=None, thresholds=[0.1, 0.3, 0.5]):
    """评估 GT contact map 的指标"""

    if dataset == 'HUMANISE':
        contact_dir = os.path.join(data_dir, 'HUMANISE/contact_motion/contacts')
        contact_files = sorted(glob(os.path.join(contact_dir, '*.npz')))
    elif dataset == 'H3D':
        contact_dir = os.path.join(data_dir, 'H3D/contacts')
        contact_files = sorted(glob(os.path.join(contact_dir, '*.npz')))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Dataset: {dataset}")
    print(f"Contact dir: {contact_dir}")
    print(f"Found files: {len(contact_files)}")

    if len(contact_files) == 0:
        print(f"ERROR: No .npz files found in {contact_dir}")
        print(f"Please check the path exists and contains .npz files")
        # 尝试列出目录内容
        if os.path.exists(contact_dir):
            files = os.listdir(contact_dir)[:10]
            print(f"Directory contents (first 10): {files}")
        else:
            print(f"Directory does not exist: {contact_dir}")
        return None

    if num_samples and num_samples < len(contact_files):
        # 随机采样
        np.random.seed(2023)
        indices = np.random.choice(len(contact_files), num_samples, replace=False)
        contact_files = [contact_files[i] for i in indices]

    print(f"Evaluating {len(contact_files)} files")
    print(f"Thresholds: {thresholds}")
    print("=" * 60)

    # 先检查第一个文件的结构
    first_file = contact_files[0]
    print(f"\nChecking first file structure: {first_file}")
    data = np.load(first_file)
    print(f"Keys in npz: {list(data.keys())}")
    for key in data.keys():
        print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    print()

    metrics = defaultdict(list)
    error_count = 0

    for f in tqdm(contact_files, desc="Evaluating GT"):
        try:
            data = np.load(f)
            dist = data['dist'].astype(np.float32)  # shape: (num_points, num_joints)

            # 如果有 obj_mask，使用它；否则用全部点
            if 'obj_mask' in data:
                obj_mask = data['obj_mask'].astype(bool)
                obj_dist = dist[obj_mask, :]
            else:
                # 没有 obj_mask，用距离最小的前10%的点作为目标区域
                min_dists = dist.min(axis=1)
                threshold_dist = np.percentile(min_dists, 10)
                obj_mask = min_dists <= threshold_dist
                obj_dist = dist[obj_mask, :]

            if obj_dist.size == 0:
                error_count += 1
                continue

            # 计算指标 (与 evaluate.py 相同的逻辑)
            dist_to_target = obj_dist.min()

            for threshold in thresholds:
                if dist_to_target < threshold:
                    metrics[f'dist_to_target_{threshold}'].append(1.0)
                else:
                    metrics[f'dist_to_target_{threshold}'].append(0.0)

            metrics['dist_to_target_average'].append(obj_dist.mean())
            metrics['dist_to_target_pelvis_average'].append(obj_dist[:, 0].mean())
            metrics['dist_to_target_min_average'].append(obj_dist.min(-1).mean())
            metrics['dist_to_target_global_min'].append(dist_to_target)

        except Exception as e:
            print(f"Error loading {f}: {e}")
            error_count += 1
            continue

    print(f"\nProcessed: {len(contact_files) - error_count}, Errors: {error_count}")

    if len(metrics['dist_to_target_global_min']) == 0:
        print("ERROR: No valid samples processed!")
        return None

    # 打印结果
    print("\n" + "=" * 60)
    print("GT Affordance Map Statistics")
    print("=" * 60)

    for key in sorted(metrics.keys()):
        values = metrics[key]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key}: {mean_val:.6f} ± {std_val:.6f}")

    print("=" * 60)

    # 额外统计
    print("\nAdditional Statistics:")
    global_mins = metrics['dist_to_target_global_min']
    print(f"  Global min distance - min: {np.min(global_mins):.4f}, max: {np.max(global_mins):.4f}")
    print(f"  Samples with min_dist < 0.1m: {sum(1 for x in global_mins if x < 0.1)} / {len(global_mins)}")
    print(f"  Samples with min_dist < 0.05m: {sum(1 for x in global_mins if x < 0.05)} / {len(global_mins)}")
    print(f"  Samples with min_dist < 0.01m: {sum(1 for x in global_mins if x < 0.01)} / {len(global_mins)}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--dataset', type=str, default='HUMANISE', choices=['HUMANISE', 'H3D'])
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate (None=all)')
    args = parser.parse_args()

    evaluate_gt_contact(args.data_dir, args.dataset, args.num_samples)
