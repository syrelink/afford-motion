"""
评估接触点位置准确性的脚本

核心指标：
1. CPLE (Contact Point Localization Error): 接触点位置误差
2. TKCO (Top-K Contact Overlap): 接触区域重叠度
3. WDE (Weighted Distance Error): 加权距离误差

用法:
    python prepare/eval_contact_localization.py \
        --data_dir data \
        --pred_dir outputs/xxx/test-xxx/HUMANISE/pred_contact \
        --dataset HUMANISE \
        --phase test
"""
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict


def contact_point_localization_error(pred_dist, gt_dist, xyz):
    """
    CPLE: 衡量接触点位置的准确性

    计算预测的接触点与GT接触点之间的欧氏距离

    Args:
        pred_dist: [N, J] 预测的距离场
        gt_dist: [N, J] GT的距离场
        xyz: [N, 3] 场景点云坐标

    Returns:
        error: 接触点位置误差 (米)
    """
    # 找到GT中距离最小的点（全局最近接触点）
    gt_min_per_point = gt_dist.min(axis=-1)  # [N]
    gt_contact_idx = gt_min_per_point.argmin()

    # 找到预测中距离最小的点
    pred_min_per_point = pred_dist.min(axis=-1)  # [N]
    pred_contact_idx = pred_min_per_point.argmin()

    # 计算两个接触点的空间距离
    gt_contact_pos = xyz[gt_contact_idx]
    pred_contact_pos = xyz[pred_contact_idx]
    error = np.linalg.norm(gt_contact_pos - pred_contact_pos)

    return error


def contact_point_localization_error_per_joint(pred_dist, gt_dist, xyz):
    """
    CPLE-J: 每个关节的接触点位置误差

    Args:
        pred_dist: [N, J] 预测的距离场
        gt_dist: [N, J] GT的距离场
        xyz: [N, 3] 场景点云坐标

    Returns:
        errors: [J] 每个关节的接触点位置误差
    """
    num_joints = gt_dist.shape[1]
    errors = []

    for j in range(num_joints):
        gt_contact_idx = gt_dist[:, j].argmin()
        pred_contact_idx = pred_dist[:, j].argmin()

        gt_contact_pos = xyz[gt_contact_idx]
        pred_contact_pos = xyz[pred_contact_idx]
        error = np.linalg.norm(gt_contact_pos - pred_contact_pos)
        errors.append(error)

    return np.array(errors)


def topk_contact_overlap(pred_dist, gt_dist, k=50):
    """
    TKCO: 衡量接触区域的重叠度

    找到GT和预测中距离最小的K个点，计算重叠比例

    Args:
        pred_dist: [N, J] 预测的距离场
        gt_dist: [N, J] GT的距离场
        k: Top-K 的 K 值

    Returns:
        overlap: 重叠比例 [0, 1]
    """
    # 每个点到最近关节的距离
    gt_min_per_point = gt_dist.min(axis=-1)  # [N]
    pred_min_per_point = pred_dist.min(axis=-1)  # [N]

    # 找到距离最小的K个点
    gt_topk_indices = np.argsort(gt_min_per_point)[:k]
    pred_topk_indices = np.argsort(pred_min_per_point)[:k]

    # 计算重叠
    gt_set = set(gt_topk_indices.tolist())
    pred_set = set(pred_topk_indices.tolist())
    overlap = len(gt_set & pred_set) / k

    return overlap


def topk_contact_overlap_multi_k(pred_dist, gt_dist, k_list=[10, 30, 50, 100]):
    """
    多个K值的TKCO
    """
    results = {}
    for k in k_list:
        results[f'TKCO@{k}'] = topk_contact_overlap(pred_dist, gt_dist, k)
    return results


def weighted_distance_error(pred_dist, gt_dist, sigma=0.2):
    """
    WDE: 加权距离误差

    对接触区域给予更高权重，衡量距离场的整体精度

    Args:
        pred_dist: [N, J] 预测的距离场
        gt_dist: [N, J] GT的距离场
        sigma: 权重衰减参数

    Returns:
        error: 加权平均误差
    """
    # 权重：距离越小（接触区域），权重越大
    gt_min_per_point = gt_dist.min(axis=-1, keepdims=True)  # [N, 1]
    weight = np.exp(-gt_min_per_point / sigma)  # [N, 1]

    # 计算逐点误差
    error = np.abs(pred_dist - gt_dist)  # [N, J]
    error_per_point = error.mean(axis=-1, keepdims=True)  # [N, 1]

    # 加权平均
    weighted_error = (error_per_point * weight).sum() / weight.sum()

    return weighted_error


def contact_region_iou(pred_dist, gt_dist, threshold=0.1):
    """
    Contact Region IoU: 接触区域的交并比

    Args:
        pred_dist: [N, J] 预测的距离场
        gt_dist: [N, J] GT的距离场
        threshold: 判定为接触的距离阈值

    Returns:
        iou: 交并比 [0, 1]
    """
    gt_min = gt_dist.min(axis=-1)
    pred_min = pred_dist.min(axis=-1)

    gt_contact_mask = gt_min < threshold
    pred_contact_mask = pred_min < threshold

    intersection = (gt_contact_mask & pred_contact_mask).sum()
    union = (gt_contact_mask | pred_contact_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def soft_localization_score(pred_dist, gt_dist, xyz, sigma=0.1):
    """
    Soft Localization Score: 软定位分数

    不是硬性判断接触点位置，而是基于高斯权重计算
    预测的接触区域与GT接触区域的对齐程度

    Args:
        pred_dist: [N, J] 预测的距离场
        gt_dist: [N, J] GT的距离场
        xyz: [N, 3] 场景点云坐标
        sigma: 高斯权重的标准差

    Returns:
        score: 定位分数 [0, 1]，越高越好
    """
    # GT的接触权重（距离越小权重越大）
    gt_min = gt_dist.min(axis=-1)
    gt_weight = np.exp(-gt_min / sigma)
    gt_weight = gt_weight / gt_weight.sum()  # 归一化

    # 预测的接触点位置
    pred_min = pred_dist.min(axis=-1)
    pred_contact_idx = pred_min.argmin()
    pred_contact_pos = xyz[pred_contact_idx]

    # 计算预测接触点到所有点的距离
    dist_to_pred = np.linalg.norm(xyz - pred_contact_pos, axis=-1)

    # 分数 = GT权重在预测接触点附近的累积
    # 如果预测接触点靠近GT接触区域，分数会很高
    alignment_weight = np.exp(-dist_to_pred / sigma)
    score = (gt_weight * alignment_weight).sum()

    return score


def evaluate_contact_localization(
    data_dir: str,
    pred_dir: str,
    dataset: str = 'HUMANISE',
    phase: str = 'test',
    use_obj_mask: bool = True
):
    """
    主评估函数

    Args:
        data_dir: 数据根目录
        pred_dir: 预测结果目录
        dataset: 数据集名称
        phase: train/test
        use_obj_mask: 是否只评估目标物体区域
    """
    # 加载 split ids
    txt_path = os.path.join(data_dir, f'{dataset}/{phase}.txt')
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found")
        return None

    with open(txt_path, 'r') as f:
        split_ids = [int(line.strip()) for line in f.readlines()]

    print(f"Dataset: {dataset}, Phase: {phase}")
    print(f"Prediction dir: {pred_dir}")
    print(f"Total samples: {len(split_ids)}")
    print(f"Use obj_mask: {use_obj_mask}")
    print("=" * 60)

    # Metrics
    metrics = defaultdict(list)

    processed = 0
    skipped = 0
    errors = 0

    for idx in tqdm(split_ids, desc="Evaluating"):
        try:
            # 加载 GT
            gt_file = os.path.join(data_dir, f'{dataset}/contact_motion/contacts/{idx:0>5}.npz')
            if not os.path.exists(gt_file):
                skipped += 1
                continue

            gt_data = np.load(gt_file)
            gt_dist = gt_data['dist'].astype(np.float32)
            xyz = gt_data['points'][:, :3].astype(np.float32)

            # 加载预测
            pred_file = os.path.join(pred_dir, f'{idx:0>5}.npy')
            if not os.path.exists(pred_file):
                skipped += 1
                continue

            pred_dist = np.load(pred_file).astype(np.float32)
            if pred_dist.ndim == 3:
                pred_dist = pred_dist[0]  # [1, N, J] -> [N, J]

            # 检查形状匹配
            if pred_dist.shape != gt_dist.shape:
                print(f"Shape mismatch at {idx}: pred {pred_dist.shape} vs gt {gt_dist.shape}")
                skipped += 1
                continue

            # 加载 obj_mask（如果需要）
            if use_obj_mask and dataset == 'HUMANISE':
                mask_file = os.path.join(data_dir, f'{dataset}/contact_motion/target_mask/{idx:0>5}.npy')
                if os.path.exists(mask_file):
                    obj_mask = np.load(mask_file)
                    gt_dist_eval = gt_dist[obj_mask]
                    pred_dist_eval = pred_dist[obj_mask]
                    xyz_eval = xyz[obj_mask]
                else:
                    gt_dist_eval = gt_dist
                    pred_dist_eval = pred_dist
                    xyz_eval = xyz
            else:
                gt_dist_eval = gt_dist
                pred_dist_eval = pred_dist
                xyz_eval = xyz

            if len(gt_dist_eval) == 0:
                skipped += 1
                continue

            # === 计算指标 ===

            # 1. CPLE: 接触点位置误差
            cple = contact_point_localization_error(pred_dist_eval, gt_dist_eval, xyz_eval)
            metrics['CPLE'].append(cple)

            # 2. CPLE-Pelvis: 骨盆接触点位置误差
            cple_joints = contact_point_localization_error_per_joint(pred_dist_eval, gt_dist_eval, xyz_eval)
            metrics['CPLE_pelvis'].append(cple_joints[0])

            # 3. TKCO: 接触区域重叠度
            tkco_results = topk_contact_overlap_multi_k(pred_dist_eval, gt_dist_eval, k_list=[10, 30, 50, 100])
            for key, value in tkco_results.items():
                metrics[key].append(value)

            # 4. WDE: 加权距离误差
            wde = weighted_distance_error(pred_dist_eval, gt_dist_eval, sigma=0.2)
            metrics['WDE'].append(wde)

            # 5. Contact Region IoU
            for thresh in [0.1, 0.2, 0.3]:
                iou = contact_region_iou(pred_dist_eval, gt_dist_eval, threshold=thresh)
                metrics[f'IoU@{thresh}'].append(iou)

            # 6. Soft Localization Score
            sls = soft_localization_score(pred_dist_eval, gt_dist_eval, xyz_eval, sigma=0.1)
            metrics['SoftLocScore'].append(sls)

            # 7. 传统指标（用于对比）
            pred_min = pred_dist_eval.min()
            gt_min = gt_dist_eval.min()
            for thresh in [0.1, 0.3, 0.5]:
                metrics[f'dist_{thresh}'].append(1.0 if pred_min < thresh else 0.0)
            metrics['min_dist_error'].append(abs(pred_min - gt_min))

            processed += 1

        except Exception as e:
            errors += 1
            print(f"Error at {idx}: {e}")

    # === 打印结果 ===
    print("\n" + "=" * 60)
    print(f"Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print("=" * 60)
    print("Contact Localization Evaluation Results")
    print("=" * 60)

    # 新指标（核心）
    print("\n--- 新指标 (接触点位置准确性) ---")
    for key in ['CPLE', 'CPLE_pelvis']:
        if key in metrics:
            values = metrics[key]
            print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f} (单位: 米)")

    print("\n--- 接触区域重叠度 (越高越好) ---")
    for key in ['TKCO@10', 'TKCO@30', 'TKCO@50', 'TKCO@100']:
        if key in metrics:
            values = metrics[key]
            print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\n--- 接触区域 IoU (越高越好) ---")
    for key in ['IoU@0.1', 'IoU@0.2', 'IoU@0.3']:
        if key in metrics:
            values = metrics[key]
            print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\n--- 其他指标 ---")
    for key in ['WDE', 'SoftLocScore']:
        if key in metrics:
            values = metrics[key]
            print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\n--- 传统指标 (用于对比) ---")
    for key in ['dist_0.1', 'dist_0.3', 'dist_0.5', 'min_dist_error']:
        if key in metrics:
            values = metrics[key]
            print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("=" * 60)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate contact point localization accuracy')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Path to prediction directory (containing xxxxx.npy files)')
    parser.add_argument('--dataset', type=str, default='HUMANISE',
                        choices=['HUMANISE', 'HumanML3D'],
                        help='Dataset name')
    parser.add_argument('--phase', type=str, default='test',
                        choices=['train', 'test'],
                        help='Data split phase')
    parser.add_argument('--no_obj_mask', action='store_true',
                        help='Do not use obj_mask for HUMANISE')
    args = parser.parse_args()

    evaluate_contact_localization(
        data_dir=args.data_dir,
        pred_dir=args.pred_dir,
        dataset=args.dataset,
        phase=args.phase,
        use_obj_mask=not args.no_obj_mask
    )
