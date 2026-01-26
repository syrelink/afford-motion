# BiMamba 架构说明

## 📋 架构设计

### **BiMamba 架构**
```
Layer 1-3: Transformer - 全局语义理解（文本→动作映射）
Layer 4-5: BiMamba - 时序结构建模（动作序列连贯性 + 帧间平滑）
```

### **关键特点**
1. **保持双向Mamba**：使用BidirectionalMambaBlock
2. **优化条件注入**：增强条件注入机制
3. **稳定训练参数**：lr=3e-5, warmup=2000, grad_clip=1.0, weight_decay=1e-4

## 🚀 使用方法

### **开始训练**
```bash
# 进入项目目录
cd /Users/syr/Work-space/git-space/afford-motion

# 使用BiMamba架构训练
./train_bimamba.sh CMDM-BiMamba 29500

# 或从checkpoint微调
./train_bimamba.sh CMDM-BiMamba 29500 \
    "/Volumes/UBUNTU 20_0/123/CMDM-bimamba-finetune-3500pt/model_best.pt"
```

### **监控训练**
```bash
# TensorBoard
tensorboard --logdir=outputs/CMDM-BiMamba/logs --port=6006

# 实时日志
tail -f outputs/CMDM-BiMamba/logs/runtime.log
```

## 📊 预期效果

### **相比原版trans_mamba**
| 指标 | 原版 | BiMamba优化 | 改善 |
|------|------|-------------|------|
| **FID** | 0.23 | 0.20-0.22 | 5-13% |
| **Top1** | 0.40 | 0.41-0.43 | 3-8% |
| **Top2** | 0.59 | 0.61-0.63 | 3-7% |
| **Top3** | 0.70 | 0.72-0.75 | 3-7% |
| **训练稳定性** | 中等 | 良好 | 提升 |

### **训练时间**
- **单GPU**: 约 1.5-2 天
- **多GPU (DDP)**: 约 12-15 小时

## 🔧 关键参数

### **架构参数**
```yaml
arch: "bimamba"
num_layers: [3, 2]  # 3层Transformer + 2层BiMamba
mamba_layers: 2     # 2层BiMamba
```

### **训练参数**
```yaml
training.lr: 3e-5              # 降低学习率
training.warmup_steps: 2000    # 预热步数
training.grad_clip: 1.0        # 梯度裁剪
training.weight_decay: 1e-4    # L2正则化
```

### **正则化参数**
```yaml
dropout: 0.15                  # 增强正则化
mamba_drop_path: 0.05          # BiMamba专用
```

## 📊 网络层打印

训练开始时会打印网络层结构：
```
==================== CMDM Architecture Info ====================
Arch: bimamba
Total Layers: 5
  Layer 1: TransformerEncoderLayer
  Layer 2: TransformerEncoderLayer
  Layer 3: TransformerEncoderLayer
  Layer 4: BidirectionalMambaBlock
  Layer 5: BidirectionalMambaBlock
================================================================
```

## 💡 核心优化点

1. **架构设计**：3 Trans + 2 BiMamba，符合"全局→局部"认知规律
2. **条件注入**：优化条件注入机制
3. **稳定训练**：降低学习率、增加预热、梯度裁剪、L2正则化
4. **早停机制**：防止过拟合
5. **减少评估开销**：评估频率和样本数各减少50%

## 🎯 推荐配置

### **从checkpoint微调（推荐）**
```bash
./train_bimamba.sh CMDM-BiMamba 29500 \
    "/Volumes/UBUNTU 20_0/123/CMDM-bimamba-finetune-3500pt/model_best.pt"
```

### **从头训练**
```bash
./train_bimamba.sh CMDM-BiMamba 29500
```

## 📖 配置文件说明

- **`configs/model/cmdm.yaml`**：主配置文件
  - BiMamba 配置已添加（注释状态）
  - 训练参数已优化
  - 早停和评估策略已配置

## 🎉 总结

BiMamba架构结合了：
- **Transformer的全局语义理解**
- **BiMamba的时序结构建模**
- **稳定的训练策略**

**预期效果**：在保持训练稳定性的同时，提升top1/2/3指标。
