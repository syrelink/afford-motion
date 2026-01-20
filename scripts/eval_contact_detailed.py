"""
详细评估 affordance map 预测质量：pred vs GT 直接对比

Usage:
    python scripts/eval_contact_detailed.py \
        --pred_dir outputs/CDM-PointMamba/eval \
        --gt_dir data/HUMANISE/contacts \
        --dataset HUMANISE
"""

import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr


def evaluate_detailed(pred_dir, gt_dir, dataset, num_samples=None):
    """详细评估 pred vs GT"""

    # 找到预测文件
    pred_files = sorted(glob(os.path.join(pred_dir, f'{dataset}/pred_contact/*.npy')))

    print(f"Pred dir: {pred_dir}")
    print(f"GT dir: {gt_dir}")
    print(f"Found {len(pred_files)} prediction files")

    if len(pred_files) == 0:
        print("ERROR: No prediction files found!")
        return None

    if num_samples and num_samples < len(pred_files):
        np.random.seed(2023)
        indices = np.random.choice(len(pred_files), num_samples, replace=False)
        pred_files = [pred_files[i] for i in indices]

    metrics = defaultdict(list)

    for pred_file in tqdm(pred_files, desc="Evaluating"):
        try:
            # 加载预测
            pred = np.load(pred_file)  # shape: (1, num_points, num_joints) or (num_points, num_joints)
            if pred.ndim == 3:
                pred = pred[0]

            # 找对应的 GT
            basename = os.path.basename(pred_file).replace('.npy', '')
            # 处理文件名格式差异
            if '-' in basename:
                basename = basename.split('-')[0]
            gt_file = os.path.join(gt_dir, f'{basename}.npz')

            if not os.path.exists(gt_file):
                continue

            gt_data = np.load(gt_file)
            gt = gt_data['dist'].astype(np.float32)

            # 确保形状匹配
            if pred.shape != gt.shape:
                # 可能需要调整
                min_points = min(pred.shape[0], gt.shape[0])
                min_joints = min(pred.shape[1], gt.shape[1])
                pred = pred[:min_points, :min_joints]
                gt = gt[:min_points, :min_joints]

            # ========== 直接对比指标 ==========

            # 1. MAE (Mean Absolute Error)
            mae = np.abs(pred - gt).mean()
            metrics['mae'].append(mae)

            # 2. MSE (Mean Squared Error)
            mse = ((pred - gt) ** 2).mean()
            metrics['mse'].append(mse)

            # 3. RMSE
            rmse = np.sqrt(mse)
            metrics['rmse'].append(rmse)

            # 4. 相关系数 (整体)
            pred_flat = pred.flatten()
            gt_flat = gt.flatten()
            if len(pred_flat) > 2:
                pearson_r, _ = pearsonr(pred_flat, gt_flat)
                metrics['pearson_r'].append(pearson_r)

            # ========== 接触区域指标 ==========

            # 5. 接触区域 IoU (threshold = 0.1m)
            pred_contact = (pred.min(axis=1) < 0.1)
            gt_contact = (gt.min(axis=1) < 0.1)
            intersection = (pred_contact & gt_contact).sum()
            union = (pred_contact | gt_contact).sum()
            iou_01 = intersection / (union + 1e-6)
            metrics['contact_iou_0.1'].append(iou_01)

            # 6. 接触区域 IoU (threshold = 0.3m)
            pred_contact = (pred.min(axis=1) < 0.3)
            gt_contact = (gt.min(axis=1) < 0.3)
            intersection = (pred_contact & gt_contact).sum()
            union = (pred_contact | gt_contact).sum()
            iou_03 = intersection / (union + 1e-6)
            metrics['contact_iou_0.3'].append(iou_03)

            # 7. 接触点的距离误差 (只看GT接触区域)
            gt_contact_mask = gt.min(axis=1) < 0.1
            if gt_contact_mask.sum() > 0:
                contact_mae = np.abs(pred[gt_contact_mask] - gt[gt_contact_mask]).mean()
                metrics['contact_region_mae'].append(contact_mae)

            # 8. 非接触区域的距离误差
            non_contact_mask = gt.min(axis=1) >= 0.3
            if non_contact_mask.sum() > 0:
                non_contact_mae = np.abs(pred[non_contact_mask] - gt[non_contact_mask]).mean()
                metrics['non_contact_region_mae'].append(non_contact_mae)

            # ========== 分布指标 ==========

            # 9. 最小距离误差
            pred_min = pred.min()
            gt_min = gt.min()
            min_dist_error = abs(pred_min - gt_min)
            metrics['min_dist_error'].append(min_dist_error)

            # 10. 预测的最小距离值
            metrics['pred_min_dist'].append(pred_min)
            metrics['gt_min_dist'].append(gt_min)

        except Exception as e:
            print(f"Error processing {pred_file}: {e}")
            continue

    # 打印结果
    print("\n" + "=" * 70)
    print("Detailed Affordance Map Evaluation: Pred vs GT")
    print("=" * 70)

    print("\n【直接对比指标】(越低越好)")
    for key in ['mae', 'mse', 'rmse']:
        if key in metrics:
            values = metrics[key]
            print(f"  {key.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\n【相关性指标】(越高越好)")
    if 'pearson_r' in metrics:
        values = metrics['pearson_r']
        print(f"  Pearson R: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\n【接触区域指标】")
    for key in ['contact_iou_0.1', 'contact_iou_0.3']:
        if key in metrics:
            values = metrics[key]
            print(f"  {key} (越高越好): {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\n【区域分离分析】")
    if 'contact_region_mae' in metrics:
        values = metrics['contact_region_mae']
        print(f"  接触区域 MAE (越低越好): {np.mean(values):.4f} ± {np.std(values):.4f}")
    if 'non_contact_region_mae' in metrics:
        values = metrics['non_contact_region_mae']
        print(f"  非接触区域 MAE (越低越好): {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\n【最小距离分析】")
    if 'min_dist_error' in metrics:
        print(f"  最小距离误差: {np.mean(metrics['min_dist_error']):.4f} ± {np.std(metrics['min_dist_error']):.4f}")
        print(f"  预测最小距离: {np.mean(metrics['pred_min_dist']):.4f} (GT: {np.mean(metrics['gt_min_dist']):.4f})")

    print("\n" + "=" * 70)

    # 总结优劣势
    print("\n【优劣势分析】")

    if 'contact_region_mae' in metrics and 'non_contact_region_mae' in metrics:
        contact_mae = np.mean(metrics['contact_region_mae'])
        non_contact_mae = np.mean(metrics['non_contact_region_mae'])

        if contact_mae < non_contact_mae:
            print(f"  ✓ 接触区域预测较好 (MAE: {contact_mae:.4f} < {non_contact_mae:.4f})")
            print(f"  ✗ 非接触区域预测较差 → 建议: 加入平滑损失")
        else:
            print(f"  ✗ 接触区域预测较差 (MAE: {contact_mae:.4f} > {non_contact_mae:.4f})")
            print(f"  → 建议: 加入 focal loss 关注接触区域")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help='Prediction directory (exp output)')
    parser.add_argument('--gt_dir', type=str, default='data/HUMANISE/contacts', help='GT directory')
    parser.add_argument('--dataset', type=str, default='HUMANISE')
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    evaluate_detailed(args.pred_dir, args.gt_dir, args.dataset, args.num_samples)
