# 训练脚本说明

## 脚本文件

- `train.sh` - 生成 CDM 第一阶段训练数据集的脚本
- `test.sh` - 测试脚本（已存在）

## 使用方法

### 基本用法

```bash
./scripts/t2m_contact/train.sh [EXP_DIR] [GPU] [SEED]
```

### 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `EXP_DIR` | Checkpoint 目录路径 | 必填 | `outputs/2026-01-18_10-30-00_cdm_train/ckpt` |
| `GPU` | GPU 设备编号 | 0 | `0` |
| `SEED` | 随机种子 | 2023 | `2023` |

### 使用示例

```bash
# 示例 1: 基本使用
./scripts/t2m_contact/train.sh outputs/2026-01-18_10-30-00_cdm_train

# 示例 2: 指定 GPU
./scripts/t2m_contact/train.sh outputs/2026-01-18_10-30-00_cdm_train 0

# 示例 3: 指定所有参数
./scripts/t2m_contact/train.sh outputs/2026-01-18_10-30-00_cdm_train 0 2023
```

## 参数详细说明

### 1. EXP_DIR (必需)
**说明**: 包含 checkpoint 文件的目录路径

**要求**:
- 目录必须存在
- 目录下必须有 `ckpt/` 子目录
- `ckpt/` 目录中必须有 `model*.pt` 文件

**示例**:
```bash
# 正确的路径
./scripts/t2m_contact/train.sh outputs/2026-01-18_10-30-00_cdm_train

# 错误的路径（会报错）
./scripts/t2m_contact/train.sh /path/to/nonexistent/dir
```

### 2. GPU (可选)
**说明**: 指定使用的 GPU 设备编号

**默认值**: `0`

**示例**:
```bash
# 使用 GPU 0
./scripts/t2m_contact/train.sh outputs/... 0

# 使用 GPU 1
./scripts/t2m_contact/train.sh outputs/... 1
```

### 3. SEED (可选)
**说明**: 随机种子，用于确保结果可复现

**默认值**: `2023`

**示例**:
```bash
# 使用默认随机种子 2023
./scripts/t2m_contact/train.sh outputs/... 0

# 使用自定义随机种子
./scripts/t2m_contact/train.sh outputs/... 0 42
```

## 脚本内部参数说明

脚本内部使用以下参数运行 `train_contact_gen.py`：

### Hydra 参数
- `hydra/job_logging=none` - 禁用 Hydra 日志
- `hydra/hydra_logging=none` - 禁用 Hydra 日志

### 基础参数
- `exp_dir=${EXP_DIR}` - Checkpoint 目录
- `seed=${SEED}` - 随机种子
- `gpu=${GPU}` - GPU 设备
- `output_dir=outputs` - 输出目录
- `exp_name=cdm_first_stage` - 实验名称

### 任务参数
- `task=contact_gen_train` - 任务类型（训练数据生成）
- `task.train.batch_size=32` - 批处理大小
- `task.train.num_workers=4` - 数据加载工作进程数
- `task.train.phase=train` - 使用训练集

### 模型参数
- `model=cdm` - CDM 模型
- `model.arch=PointMamba` - 使用 ContactPointMamba 架构
- `model.scene_model.use_scene_model=False` - 不使用场景模型
- `model.text_model.max_length=20` - 文本最大长度

### 数据集参数
- `task.dataset.sigma=0.8` - 高斯核标准差
- `task.dataset.num_points=8192` - 点云数量
- `task.dataset.use_color=true` - 使用颜色特征
- `task.dataset.use_openscene=false` - 不使用 OpenScene

### 扩散参数
- `diffusion.steps=1000` - 扩散步数
- `diffusion.noise_schedule=cosine` - 噪声调度

## 输出文件

运行后会在 `outputs/{timestamp}_cdm_first_stage/train-{timestamp}/` 目录中生成：

```
train-{timestamp}/
├── train.log                      # 运行日志
├── checkpoint_info.txt            # Checkpoint 信息
├── generated_contact_maps.npz     # 生成的训练数据（最终文件）
└── temp_batch_*.npz               # 临时进度文件（运行后删除）
```

## 常见问题

### 1. 找不到 checkpoint

**错误信息**:
```
错误: 找不到 checkpoint 目录 /path/to/exp_dir/ckpt
```

**解决方法**:
```bash
# 检查 checkpoint 目录
ls -la /path/to/exp_dir/ckpt/

# 确保路径正确
./scripts/t2m_contact/train.sh /path/to/correct/exp_dir
```

### 2. GPU 内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方法**:
```bash
# 修改脚本中的 batch_size
# 将 task.train.batch_size=32 改为 task.train.batch_size=16
```

### 3. 数据集路径错误

**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方法**:
```bash
# 检查数据集路径
ls -la data/H3D/

# 修改配置文件中的数据集路径
```

## 与 test.sh 的对比

| 特性 | test.sh | train.sh |
|------|---------|----------|
| 主要用途 | 模型测试 | 训练数据生成 |
| 输入参数 | EXP_DIR, EVAL_MODE, SEED | EXP_DIR, GPU, SEED |
| 运行脚本 | test.py | train_contact_gen.py |
| 输出结果 | 评估指标 | 生成的接触地图数据 |
| 自动保存 | 无 | 每 10 个 batch 保存 |

## 技术细节

### ContactPointMamba 架构

CDM 第一阶段使用 ContactPointMamba 架构，主要特点：

1. **Hilbert 空间排序**：将点云按照 Hilbert 曲线排序
2. **双向 Mamba 扫描**：正向和反向扫描点云序列
3. **共享权重**：正反向使用相同的 Mamba 权重，节省显存
4. **残差连接**：正确融合正反向信息

### 数据生成流程

1. **加载数据集**：ContactMapDataset
2. **加载模型**：CDM + ContactPointMamba
3. **加载 checkpoint**：训练好的模型权重
4. **生成接触地图**：使用扩散模型采样
5. **保存数据**：自动保存到输出目录

## 示例输出

```
==========================================
CDM 第一阶段训练数据生成
==========================================
Checkpoint 目录: outputs/2026-01-18_10-30-00_cdm_train
GPU: 0
随机种子: 2023
==========================================

[Configuration]
...

[Train Contact Gen] ==> Begin generating training data..
Load train dataset size: 1234
Load checkpoint from outputs/.../ckpt/model001000.pt

=== Checkpoint Info ===
Checkpoint: outputs/.../ckpt/model001000.pt
Output directory: outputs/.../train-...
Dataset size: 1234
Batch size: 32

batch index: 0, is k_sample_batch: False, case index: 12345
Auto-saved progress to outputs/.../train-.../temp_batch_000010.npz
...

=== Summary ===
Total 1234 samples generated
Output directory: outputs/.../train-...
Final data file: outputs/.../train-.../generated_contact_maps.npz

==========================================
训练数据生成完成!
==========================================
```

## 下一步

1. **查看生成的数据**：
   ```bash
   ls -la outputs/*/train-*/
   ```

2. **查看详细文档**：
   - `../../README_TRAIN_CONTACT_GEN.md` - 详细使用说明

3. **使用生成的数据**：
   - 训练 CDM 第二阶段模型
   - 数据分析和可视化
   - 模型评估和对比
