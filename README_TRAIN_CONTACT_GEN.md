# CDM 第一阶段训练数据生成

## 概述

`train_contact_gen.py` 是一个用于生成 CDM (Contact Diffusion Model) 第一阶段训练数据集的脚本。它基于 `test.py` 修改，将测试模式改为训练模式，用于生成接触地图数据。

## 使用方法

### 基本命令

```bash
cd /Users/syr/Work-space/git-space/afford-motion

# 基本使用（使用默认配置）
python train_contact_gen.py

# 指定 GPU
python train_contact_gen.py gpu=0

# 指定实验名称
python train_contact_gen.py exp_name=cdm_first_stage

# 指定 batch size
python train_contact_gen.py task.train.batch_size=32
```

### 完整命令示例

```bash
# 使用 ContactPointMamba 架构生成训练数据
python train_contact_gen.py \
    gpu=0 \
    exp_name=cdm_first_stage \
    task=contact_gen_train \
    task.train.batch_size=32 \
    model=cdm \
    model.arch=PointMamba
```

## 输出文件

运行后会在 `outputs/{timestamp}_{exp_name}/train-{timestamp}/` 目录中生成：

```
train-{timestamp}/
├── train.log                      # 运行日志
├── checkpoint_info.txt            # Checkpoint 信息
├── generated_contact_maps.npz     # 生成的训练数据（最终文件）
└── temp_batch_*.npz               # 临时进度文件（运行后删除）
```

### checkpoint_info.txt 内容

```
Checkpoint: outputs/.../ckpt/model001000.pt
Output directory: outputs/.../train-...
Dataset size: 12345
Batch size: 32

Configuration:
[完整的配置信息]
```

## 主要特性

1. **自动保存**：每 10 个 batch 自动保存一次进度，防止数据丢失
2. **输出路径显示**：明确显示输出路径和输入的 checkpoint
3. **进度跟踪**：实时显示处理p进度和 checkoint 信息
4. **详细日志**：完整的运行日志和配置信息记录

## 配置说明

### 数据集配置

```yaml
dataset:
  name: 'ContactMapDataset'  # CDM 第一阶段数据集
  data_dir: './data'         # 数据集路径
  sets: ['HumanML3D', 'HUMANISE', 'PROX']  # 使用的数据集
  num_points: 8192           # 点云数量
  use_color: true            # 使用颜色特征
```

### 模型配置

```yaml
model:
  name: CDM                  # CDM 模型
  arch: PointMamba          # 使用 ContactPointMamba 架构
  data_repr: 'contact_cont_joints'  # 数据表示方式
```

## 常见问题

### 1. 找不到 checkpoint

```bash
# 检查 checkpoint 目录
ls -la outputs/your_exp_dir/ckpt/

# 指定正确的 exp_dir
python train_contact_gen.py exp_dir=/path/to/checkpoint
```

### 2. 数据集路径错误

```bash
# 检查数据集
ls -la data/H3D/

# 修改配置
python train_contact_gen.py dataset.data_dir=/path/to/data
```

### 3. GPU 内存不足

```bash
# 减小 batch size
python train_contact_gen.py task.train.batch_size=16

# 或使用 CPU
python train_contact_gen.py gpu=null
```

### 4. 程序意外中断

- 脚本每 10 个 batch 自动保存一次进度
- 检查 `temp_batch_*.npz` 文件
- 可以从最近的临时文件恢复数据

## 与 test.py 的对比

| 特性 | test.py | train_contact_gen.py |
|------|---------|---------------------|
| 主要用途 | 模型测试 | 训练数据生成 |
| 数据集阶段 | 'test' | 'train' |
| 输出结果 | 评估指标 | 生成的接触地图数据 |
| 自动保存 | 无 | 每 10 个 batch 保存 |
| 进度显示 | 基本信息 | 详细 checkpoint 信息 |

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
```

## 下一步

生成的训练数据可以用于：
1. 训练 CDM 第二阶段模型
2. 数据分析和可视化
3. 模型评估和对比
