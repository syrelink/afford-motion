"""
评估 GT affordance map 的脚本
完全按照 utils/evaluate.py 中 ContactEvaluator.evaluate() 的逻辑来评估

用法:
    python eval_gt_contact.py --data_dir data --dataset HUMANISE --phase test
"""
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def eval_gt_contact(data_dir, dataset='HUMANISE', phase='test'):
    """
    按照 ContactEvaluator.evaluate() 的评估逻辑评估 GT

    关键点（与 utils/evaluate.py 第 177-189 行完全一致）：
    1. 使用 info_obj_mask 过滤目标物体的点
    2. obj_dist = dist[obj_mask, :]
    3. dist_to_target = obj_dist.min()
    """
    # 加载 split ids
    txt_path = os.path.join(data_dir, f'{dataset}/{phase}.txt')
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found")
        return None

    with open(txt_path, 'r') as f:
        split_ids = [int(line.strip()) for line in f.readlines()]

    print(f"Dataset: {dataset}, Phase: {phase}")
    print(f"Total samples: {len(split_ids)}")

    # 阈值（与 ContactEvaluator 配置一致）
    thresholds = [0.1, 0.3, 0.5]

    # metrics
    metrics = defaultdict(list)

    processed = 0
    skipped = 0
    errors = 0

    for idx in tqdm(split_ids, desc="Evaluating GT"):
        try:
            # 加载 GT contact (与 ContactMapDataset.__getitem__ 一致)
            cont_file = os.path.join(data_dir, f'{dataset}/contact_motion/contacts/{idx:0>5}.npz')
            if not os.path.exists(cont_file):
                skipped += 1
                continue

            contact = np.load(cont_file)
            dist = contact['dist'].astype(np.float32)  # (num_points, num_joints)

            # 加载 obj_mask（与 ContactMapDataset 第 653-658 行一致）
            # 只有 HUMANISE 数据集有 target_mask
            if dataset == 'HUMANISE':
                target_mask_file = os.path.join(data_dir, f'{dataset}/contact_motion/target_mask/{idx:0>5}.npy')
                if not os.path.exists(target_mask_file):
                    skipped += 1
                    continue
                obj_mask = np.load(target_mask_file)
            else:
                # 其他数据集 info_obj_mask = None，评估逻辑会跳过
                # 这里我们使用所有点进行评估
                obj_mask = np.ones(dist.shape[0], dtype=bool)

            # 只取目标物体的点（与 ContactEvaluator.evaluate 第 178-179 行一致）
            obj_dist = dist[obj_mask, :]

            if obj_dist.shape[0] == 0:
                skipped += 1
                continue

            # 计算指标（与 ContactEvaluator.evaluate 第 181-189 行完全一致）
            dist_to_target = obj_dist.min()  # 全局最小距离

            for threshold in thresholds:
                if dist_to_target < threshold:
                    metrics[f'dist_to_target_{threshold}'].append(1.0)
                else:
                    metrics[f'dist_to_target_{threshold}'].append(0.0)

            metrics['dist_to_target_average'].append(obj_dist.mean())
            metrics['dist_to_target_pelvis_average'].append(obj_dist[:, 0].mean())
            metrics['dist_to_target_min_average'].append(obj_dist.min(-1).mean())
            metrics['dist_to_target_global_min'].append(dist_to_target)

            processed += 1

        except Exception as e:
            errors += 1
            print(f"Error processing {idx}: {e}")

    # 打印结果
    print(f"\n{'='*60}")
    print(f"Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"{'='*60}")
    print("GT Affordance Map Evaluation Results")
    print(f"{'='*60}")

    # 按照与模型评估相同的顺序打印
    for threshold in thresholds:
        key = f'dist_to_target_{threshold}'
        if key in metrics:
            values = metrics[key]
            mean = np.mean(values)
            std = np.std(values)
            print(f"{key}: {mean:.6f} ± {std:.6f}")

    for key in ['dist_to_target_average', 'dist_to_target_min_average',
                'dist_to_target_pelvis_average', 'dist_to_target_global_min']:
        if key in metrics:
            values = metrics[key]
            mean = np.mean(values)
            std = np.std(values)
            print(f"{key}: {mean:.6f} ± {std:.6f}")

    print(f"{'='*60}")

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate GT affordance map')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='HUMANISE',
                        choices=['HUMANISE', 'HumanML3D'],
                        help='Dataset name')
    parser.add_argument('--phase', type=str, default='test',
                        choices=['train', 'test'],
                        help='Data split phase')
    args = parser.parse_args()

    eval_gt_contact(args.data_dir, args.dataset, args.phase)
