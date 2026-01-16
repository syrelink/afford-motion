# 1. 设置后端
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
import pyrender
import cv2
import glob
import sys
import traceback
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 👉 配置区
# ==========================================
SOURCE_DIR = "outputs/CDM-Perceiver-HUMANISE-step200k/eval/test-1127-204318/saved_samples"
OUTPUT_DIR = "outputs/CDM-Perceiver-HUMANISE-step200k/eval/test-1127-204318/map-mesh-2k-contrast"
DATASET_ROOT = "/home/supermicro/syr/git-sapce/afford-motion"

# ⭐ 限制设置
MAX_PER_ACTION = 5  # 每个动作文本最多生成几张？
MAX_Total_IMAGES = None


# ==========================================

def smart_path_join(root, rel_path):
    if rel_path.startswith('./'): rel_path = rel_path[2:]
    root = root.rstrip('/')
    if root.endswith('data') and rel_path.startswith('data/'):
        rel_path = rel_path[5:]
    return os.path.join(root, rel_path)


def load_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        raise RuntimeError(f"无法加载 NPY 文件: {e}")

    pc_xyz = None
    for k in ['c_pc_xyz', 'c_pc_scene', 'x', 'points']:
        if k in data:
            pc_xyz = data[k]
            break

    mesh_rel_path = data.get('info_scene_mesh')
    trans_matrix = data.get('info_scene_trans')
    scores = data.get('sample')

    raw_text = data.get('c_text', data.get('text', ["Unknown"]))
    text = str(raw_text[0]) if isinstance(raw_text, (list, np.ndarray)) and len(raw_text) > 0 else str(raw_text)
    if isinstance(text, bytes): text = text.decode('utf-8')
    clean_text = " ".join(text.split())

    if scores is not None and pc_xyz is not None:
        flat = scores.flatten()
        if flat.size != pc_xyz.shape[0]:
            if flat.size % pc_xyz.shape[0] == 0:
                channels = flat.size // pc_xyz.shape[0]
                scores = flat.reshape(pc_xyz.shape[0], channels).max(axis=1)
        else:
            scores = flat

    return pc_xyz, scores, mesh_rel_path, trans_matrix, clean_text


def get_heatmap_colors(scores):
    vmin, vmax = np.percentile(scores, 2), np.percentile(scores, 98)
    if vmax - vmin < 1e-5: vmax = vmin + 1e-5
    norm = np.clip((scores - vmin) / (vmax - vmin), 0, 1)

    _map = np.uint8(255 * norm)
    bgr = cv2.applyColorMap(_map, cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    return rgb / 255.0, norm


def project_heatmap_to_mesh_smooth(mesh, pc_xyz, pc_colors, pc_scores_norm, k=3):
    tree = cKDTree(pc_xyz)
    dists, indices = tree.query(mesh.vertices, k=k)

    weights = 1.0 / (dists + 1e-6) ** 2
    weights /= np.sum(weights, axis=1, keepdims=True)

    neighbor_colors = pc_colors[indices]
    interpolated_colors = np.sum(neighbor_colors * weights[:, :, np.newaxis], axis=1)

    neighbor_scores = pc_scores_norm[indices]
    interpolated_scores = np.sum(neighbor_scores * weights, axis=1)

    # ✨ 关键修改 1：加深底色
    # 从 0.65 改为 0.5。颜色越深，在白背景下轮廓越清楚，也不会反光太强。
    base_color = np.array([0.5, 0.5, 0.5])

    w = interpolated_scores[:, np.newaxis]
    w = np.power(w, 0.6)

    final_colors = interpolated_colors * w + base_color * (1 - w)

    min_dist = dists[:, 0]
    mask = min_dist > 0.1
    final_colors[mask] = base_color

    return final_colors


def render_mesh_2k(mesh, text, save_path):
    # ✨ 关键修改 2：降低灯光强度，防止过曝
    # ambient_light 从 0.7 降为 0.5
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])

    py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(py_mesh)

    center = mesh.bounds.mean(axis=0)
    max_dim = np.max(mesh.extents)

    camera_eye = center + np.array([0, -max_dim * 1.1, max_dim * 0.9])
    forward = center - camera_eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 0, 1]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_eye

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=pose)

    light_pose = pose.copy()
    light_pose[:3, 3] += np.array([2, 5, 5])
    # 主光强度从 3.0 降为 2.0
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light_pose)

    fill_light_pose = pose.copy()
    fill_light_pose[:3, 3] += np.array([-2, -2, 5])
    # 补光强度从 1.5 降为 1.0
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 0.9], intensity=1.0), pose=fill_light_pose)

    # ✨ 关键修改 3：分辨率调整为 2K (2048)
    RENDER_RES = 4096  # 内部渲染分辨率 (为了抗锯齿)
    TARGET_RES = 2048  # 最终输出分辨率

    try:
        r = pyrender.OffscreenRenderer(RENDER_RES, RENDER_RES)
        color, _ = r.render(scene)
        r.delete()
    except Exception as e:
        print(f"⚠️ 显存不足，自动降级渲染: {e}")
        r = pyrender.OffscreenRenderer(2048, 2048)
        color, _ = r.render(scene)
        r.delete()
        TARGET_RES = 1024

    img = Image.fromarray(color)
    draw = ImageDraw.Draw(img)

    font_size = int(TARGET_RES * 0.05)
    try:
        font_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arial.ttf"]
        font = next((ImageFont.truetype(p, font_size) for p in font_paths if os.path.exists(p)),
                    ImageFont.load_default())
    except:
        font = ImageFont.load_default()

    draw.text((font_size, font_size), f"Action: {text}", font=font, fill=(50, 50, 50))

    # 缩小采样，获得平滑边缘
    if img.size[0] != TARGET_RES:
        img = img.resize((TARGET_RES, TARGET_RES), Image.LANCZOS)

    img.save(save_path, quality=95, optimize=True)


def is_valid_mesh_file(path):
    if not os.path.exists(path): return False, "不存在"
    if not os.path.isfile(path): return False, "是目录"
    if os.path.getsize(path) == 0: return False, "空文件"
    if path.endswith('.ply'):
        try:
            with open(path, 'rb') as f:
                if f.read(4) != b'ply\n': return False, "PLY头错误"
        except:
            return False, "读取错误"
    return True, "有效"


def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 源文件夹不存在: {SOURCE_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.npy")))

    if len(files) == 0:
        print("❌ 源文件夹为空")
        return

    if MAX_Total_IMAGES is not None: files = files[:MAX_Total_IMAGES]

    print(f"🚀 开始 2K 高对比度渲染 (每种动作限制 {MAX_PER_ACTION} 张)...")
    print(f"📂 数据集根目录: {DATASET_ROOT}")

    action_counter = {}

    for i, f in enumerate(files):
        file_id = os.path.splitext(os.path.basename(f))[0]
        save_path = os.path.join(OUTPUT_DIR, f"{file_id}.png")

        try:
            pc_xyz, scores, mesh_rel_path, trans, text = load_data(f)

            current_count = action_counter.get(text, 0)

            if current_count >= MAX_PER_ACTION:
                # print(f"⏩ [跳过] {text[:20]}... (已满 {MAX_PER_ACTION} 张)", end="\r")
                continue

            if mesh_rel_path is None: continue

            mesh_full_path = smart_path_join(DATASET_ROOT, mesh_rel_path)
            valid, reason = is_valid_mesh_file(mesh_full_path)

            if not valid:
                print(f"\n⚠️ 跳过坏文件 {file_id}: {reason}")
                continue

            print(f"🎨 [{i + 1}/{len(files)}] 正在渲染: {text} (第 {current_count + 1} 张)")

            mesh = trimesh.load(mesh_full_path, force='mesh', process=False)
            if trans is not None: mesh.apply_transform(trans)

            pc_colors, pc_scores_norm = get_heatmap_colors(scores)

            # 使用更深的底色
            mesh.visual.vertex_colors = project_heatmap_to_mesh_smooth(mesh, pc_xyz, pc_colors, pc_scores_norm, k=3)

            # 使用 2K 渲染
            render_mesh_2k(mesh, text, save_path)

            action_counter[text] = current_count + 1

        except Exception as e:
            print(f"\n❌ 处理 {file_id} 错误: {e}")
            continue

    print(f"\n\n✨ 全部完成! 统计如下:")
    for action, count in action_counter.items():
        print(f" - {action}: 生成 {count} 张")
    print(f"📂 结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()