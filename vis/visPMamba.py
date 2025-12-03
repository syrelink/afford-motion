import os
import numpy as np
import open3d as o3d

# ========= 这里改成你自己的 PLY 文件路径 =========
PLY_PATH = "/home/supermicro/zh/BFANet-master/log/ScanNetv2/octformer/epoch400_spherecrop12w_1e-3/pred_ply/scene0011_00_pred.ply"
# ===============================================

# PointMamba 里常见的 patch 数量级，这里你可以自己调
NUM_PATCHES = 512   # 对应 n
# 邻域点数 k 在这里用“最近中心分配”来近似，不单独取 KNN 补点了


def fps_downsample(points, num_patches):
    """
    使用 Open3D 的最远点采样 (FPS)，模拟 PointMamba 中的 key points 选择。
    输入:
        points: (N, 3)
        num_patches: 采样的中心数 n
    输出:
        center_points: (n, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    num_patches = min(num_patches, points.shape[0])
    centers_pcd = pcd.farthest_point_down_sample(num_patches)
    center_points = np.asarray(centers_pcd.points)  # (n, 3)
    return center_points


def assign_points_to_centers(points, center_points):
    """
    用 KDTree 把每个点分配到最近的中心，近似 PointMamba 的 patch 归属。
    输入:
        points: (N, 3)
        center_points: (n, 3)
    输出:
        assign_idx: (N,) 每个点对应的中心编号 [0, n-1]
    """
    centers_pcd = o3d.geometry.PointCloud()
    centers_pcd.points = o3d.utility.Vector3dVector(center_points)
    kdtree = o3d.geometry.KDTreeFlann(centers_pcd)

    N = points.shape[0]
    assign_idx = np.zeros(N, dtype=np.int32)

    for i in range(N):
        # k=1 最近邻
        _, idx, _ = kdtree.search_knn_vector_3d(points[i], 1)
        assign_idx[i] = idx[0]

    return assign_idx


def simulate_pointmamba_encoder_features(points_norm, center_points, assign_idx):
    """
    模拟 PointMamba encoder 的输出特征 Z_N ∈ R^{n × C}（这里 C=3，用作颜色）。

    做法：
      - 对每个 patch（一个中心 + 其归属点）：
          1. 计算相对坐标：patch_pts - center
          2. 取 |相对坐标| 的均值，得到一个 3 维向量，作为该 token 的“几何特征”
      - 把所有 token 特征按维度归一化到 [0,1]，用于上色。

    输入:
        points_norm: (N, 3) 归一化后的点
        center_points: (n, 3)
        assign_idx: (N,)
    输出:
        token_colors: (n, 3) 归一化后的“encoder 特征”，可直接作为 RGB
    """
    n = center_points.shape[0]
    token_feats = np.zeros((n, 3), dtype=np.float32)

    for cid in range(n):
        mask = (assign_idx == cid)
        patch_pts = points_norm[mask]
        if patch_pts.shape[0] == 0:
            # 该中心下面没有点，直接用中心坐标的绝对值当特征
            rel = np.abs(center_points[cid:cid+1, :])
        else:
            rel = patch_pts - center_points[cid : cid + 1]  # (m, 3)
            rel = np.abs(rel)  # 只取绝对值，表示“局部尺度”
        feat = rel.mean(axis=0)  # (3,)
        token_feats[cid] = feat

    # 归一化到 [0,1]
    f_min = token_feats.min(axis=0, keepdims=True)
    f_max = token_feats.max(axis=0, keepdims=True)
    denom = np.maximum(f_max - f_min, 1e-6)
    token_colors = (token_feats - f_min) / denom  # (n, 3)

    return token_colors


def main():
    ply_path = PLY_PATH

    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    print(f"[Open3D] Loading PLY: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    pts = np.asarray(pcd.points)  # (N, 3)
    print(f"[Open3D] points shape: {pts.shape}")

    if pts.shape[0] == 0:
        raise ValueError("点云为空，无法可视化")

    # ========= Step 1: 归一化到单位球（模拟 PointMamba 的预处理） =========
    center = pts.mean(axis=0, keepdims=True)
    pts_centered = pts - center
    scale = np.max(np.linalg.norm(pts_centered, axis=1))
    scale = max(scale, 1e-6)
    pts_norm = pts_centered / scale  # (N, 3)
    print("[Info] 点云已归一化到单位球")

    # ========= Step 2: FPS 采样 key points（模拟 n 个 patch） =========
    print(f"[Info] 使用 FPS 采样 {NUM_PATCHES} 个中心点（key points）...")
    center_points = fps_downsample(pts_norm, NUM_PATCHES)
    n = center_points.shape[0]
    print(f"[Info] 实际采样中心点数: {n}")

    # ========= Step 3: 为每个点分配最近中心（模拟 patch 归属） =========
    print("[Info] 构建 KDTree，进行最近中心分配 ...")
    assign_idx = assign_points_to_centers(pts_norm, center_points)
    print("[Info] patch 分配完成")

    # ========= Step 4: 模拟 PointMamba encoder 输出特征，并映射为颜色 =========
    print("[Info] 模拟 PointMamba encoder 输出特征（token-level） ...")
    token_colors = simulate_pointmamba_encoder_features(pts_norm, center_points, assign_idx)
    # 把 token 颜色扩展到每个点
    point_colors = token_colors[assign_idx]  # (N, 3)

    # ========= Step 5: 把颜色写回点云并展示 =========
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="PointMamba-like Encoder Visualization",
        width=1600,
        height=1200
    )
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = 1.0          # 想让点更粗可以改大，比如 2.0
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()