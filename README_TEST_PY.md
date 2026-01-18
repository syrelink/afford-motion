# test.py 说明文档

## 文件概述

`test.py` 是 CDM (Contact Diffusion Model) 项目的测试脚本，用于评估模型性能和生成测试数据集。

## test.py 的含义

### 主要功能
1. **模型测试**：使用训练好的模型在测试集上进行评估
2. **性能评估**：计算各种评估指标（接触准确率、位置误差等）
3. **数据生成**：生成接触地图（contact map）用于后续分析

### 核心组件

#### 1. `test()` 函数
**用途**：测试模型性能
**输入**：测试数据集
**输出**：评估指标和测试结果
**特点**：
- 使用测试集（phase='test'）
- 计算评估指标
- 生成测试报告

#### 2. `train()` 函数（新增）
**用途**：生成 map_affordance 测试数据集
**输入**：训练数据集
**输出**：生成的接触地图数据
**特点**：
- 使用训练集（phase='train'）
- 不计算评估指标
- 保存生成的数据

## test() 函数详解

### 函数签名
```python
def test(cfg: DictConfig) -> None:
```

### 必要位置中文注释

#### 1. 创建输出目录（第24-25行）
```python
# 创建测试输出目录，格式：eval/test-{时间戳}
test_dir = os.path.join(cfg.eval_dir, 'test-' + time_str(Y=False))
mkdir_if_not_exists(test_dir)
```

#### 2. 添加日志记录器（第26-28行）
```python
# 添加日志记录器
logger.add(os.path.join(test_dir, 'test.log'))
logger.info('[Configuration]\\n' + OmegaConf.to_yaml(cfg) + '\\n')
logger.info('[Test] ==> Begin testing..')
```

#### 3. 设置设备（第30-33行）
```python
# 设置设备（GPU或CPU）
if cfg.gpu is not None:
    device = f'cuda:{cfg.gpu}'
else:
    device = 'cpu'
```

#### 4. 准备测试数据集（第35-39行）
```python
# 准备测试数据集
# 支持通过配置指定phase，默认为'test'
phase = cfg.task.dataset.get('phase', 'test')
test_dataset = create_dataset(cfg.task.dataset, phase, gpu=cfg.gpu, **cfg.task.test)
logger.info(f'Load {phase} dataset size: {len(test_dataset)}')
```

#### 5. 创建测试数据加载器（第41-47行）
```python
# 创建测试数据加载器
test_dataloader = test_dataset.get_dataloader(
    batch_size=cfg.task.test.batch_size,
    collate_fn=collate_fn_general,
    num_workers=cfg.task.test.num_workers,
    pin_memory=True,
    shuffle=False,  # 测试时不需要打乱数据
)
```

#### 6. 创建模型和扩散过程（第49-51行）
```python
## 创建模型和扩散过程
model, diffusion = create_model_and_diffusion(cfg, device=device)
model.to(device)
```

#### 7. 加载checkpoint（第53-56行）
```python
# 加载checkpoint
ckpts = natsorted(glob.glob(os.path.join(cfg.exp_dir, 'ckpt', 'model*.pt')))
assert len(ckpts) > 0, 'No checkpoint found.'
load_ckpt(model, ckpts[-1])
logger.info(f'Load checkpoint from {ckpts[-1]}')
```

#### 8. 创建评估器（第58-59行）
```python
## 创建评估器
evaluator = create_evaluator(cfg.task, device=device)
```

#### 9. 设置采样函数（第61-63行）
```python
## 采样（生成）
model.eval()  # 设置为评估模式
sample_fn = diffusion.p_sample_loop  # 使用扩散模型的采样函数
```

#### 10. 确定k_samples批次（第65-73行）
```python
B = test_dataloader.batch_size
sample_list = []
k_samples_list = []

# 确定哪些batch需要生成多个样本（k_samples）
if evaluator.k_samples > 0:
    k_samples_idxs = list(range(
        evaluator.num_k_samples // B))  # 前len(k_samples_idxs)个batch用于生成k个样本
else:
    k_samples_idxs = []
logger.info(f'k_samples_idxs: {k_samples_idxs}')
```

#### 11. 遍历测试数据（第75-141行）
```python
# 遍历测试数据
for i, data in enumerate(test_dataloader):
    logger.info(f\"batch index: {i}, is k_sample_batch: {i in k_samples_idxs}, case index: {data['info_index']}\")
    x = data['x']  # 输入数据（接触地图）

    # 准备模型输入参数
    x_kwargs = {}
    if 'x_mask' in data:
        x_kwargs['x_mask'] = data['x_mask'].to(device)

    # 将所有c_和info_开头的键添加到x_kwargs
    for key in data:
        if key.startswith('c_') or key.startswith('info_'):
            if torch.is_tensor(data[key]):
                x_kwargs[key] = data[key].to(device)
            else:
                x_kwargs[key] = data[key]

    use_k_sample = i in k_samples_idxs
    repeat_times = evaluator.k_samples if use_k_sample else 1

    sample_list_np = []
    k_samples_list_np = []

    # 生成样本（可能需要重复多次）
    for k in range(repeat_times):
        if cfg.model.name.startswith('CMDM'):
            ## 如果使用CMDM模型，输入c_pc_contact包含k个样本
            ## 需要移除这个项，并使用第k个接触地图
            x_kwargs['c_pc_contact'] = data['c_pc_contact'][:, k, :, :].to(device)

        # 使用扩散模型生成样本
        sample = sample_fn(
            model,
            x.shape,
            clip_denoised=False,
            noise=None,
            model_kwargs=x_kwargs,
            progress=True,
        )

        if k == 0:
            for bsi in range(B):
                sample_list_np.append(sample[bsi].cpu().numpy())

        if use_k_sample:
            for bsi in range(B):
                k_samples_list_np.append(sample[bsi].cpu().numpy())

    ## 保存单个样本
    for bsi in range(B):
        res_dict = {'sample': sample_list_np[bsi]}
        for key in data:
            if torch.is_tensor(data[key]):
                res_dict[key] = data[key][bsi].cpu().numpy()
            else:
                res_dict[key] = data[key][bsi]
        sample_list.append(res_dict)

    ## 保存k个样本
    if use_k_sample:
        for bsi in range(B):
            res_dict = {'k_samples': np.stack(k_samples_list_np[bsi::B])}
            for key in data:
                if torch.is_tensor(data[key]):
                    res_dict[key] = data[key][bsi].cpu().numpy()
                else:
                    res_dict[key] = data[key][bsi]
            k_samples_list.append(res_dict)

    ## 如果达到最大样本数，停止评估
    if i + 1 >= evaluator.eval_nbatch:
        break
```

#### 12. 计算评估指标（第143-145行）
```python
## 计算评估指标
evaluator.evaluate(sample_list, k_samples_list, test_dir, test_dataloader, device=device)
evaluator.report(test_dir)
```

## train() 函数详解

### 函数签名
```python
def train(cfg: DictConfig) -> None:
```

### 必要位置中文注释

#### 1. 创建输出目录（第154-156行）
```python
# 创建训练输出目录，格式：eval/train-{时间戳}
train_dir = os.path.join(cfg.eval_dir, 'train-' + time_str(Y=False))
mkdir_if_not_exists(train_dir)
```

#### 2. 添加日志记录器（第158-161行）
```python
# 添加日志记录器
logger.add(os.path.join(train_dir, 'train.log'))
logger.info('[Configuration]\\n' + OmegaConf.to_yaml(cfg) + '\\n')
logger.info('[Train] ==> Begin generating map_affordance test dataset..')
```

#### 3. 设置设备（第163-167行）
```python
# 设置设备（GPU或CPU）
if cfg.gpu is not None:
    device = f'cuda:{cfg.gpu}'
else:
    device = 'cpu'
```

#### 4. 准备训练数据集（第169-173行）
```python
# 准备训练数据集（用于生成测试数据）
# 使用训练阶段的数据，但用于生成测试数据集
phase = cfg.task.dataset.get('phase', 'train')
train_dataset = create_dataset(cfg.task.dataset, phase, gpu=cfg.gpu, **cfg.task.train)
logger.info(f'Load {phase} dataset size: {len(train_dataset)}')
```

#### 5. 创建训练数据加载器（第175-182行）
```python
# 创建训练数据加载器
train_dataloader = train_dataset.get_dataloader(
    batch_size=cfg.task.train.batch_size,
    collate_fn=collate_fn_general,
    num_workers=cfg.task.train.num_workers,
    pin_memory=True,
    shuffle=False,  # 生成时不需要打乱数据
)
```

#### 6. 创建模型和扩散过程（第184-186行）
```python
## 创建模型和扩散过程
model, diffusion = create_model_and_diffusion(cfg, device=device)
model.to(device)
```

#### 7. 加载checkpoint（第188-192行）
```python
# 加载checkpoint
ckpts = natsorted(glob.glob(os.path.join(cfg.exp_dir, 'ckpt', 'model*.pt')))
assert len(ckpts) > 0, 'No checkpoint found.'
load_ckpt(model, ckpts[-1])
logger.info(f'Load checkpoint from {ckpts[-1]}')
```

#### 8. 创建评估器（第194-195行）
```python
## 创建评估器
evaluator = create_evaluator(cfg.task, device=device)
```

#### 9. 设置采样函数（第197-199行）
```python
## 采样（生成map_affordance）
model.eval()  # 设置为评估模式
sample_fn = diffusion.p_sample_loop  # 使用扩散模型的采样函数
```

#### 10. 确定k_samples批次（第201-211行）
```python
B = train_dataloader.batch_size
sample_list = []
k_samples_list = []

# 确定哪些batch需要生成多个样本（k_samples）
if evaluator.k_samples > 0:
    k_samples_idxs = list(range(
        evaluator.num_k_samples // B))  # 前len(k_samples_idxs)个batch用于生成k个样本
else:
    k_samples_idxs = []
logger.info(f'k_samples_idxs: {k_samples_idxs}')
```

#### 11. 遍历训练数据（第213-285行）
```python
# 遍历训练数据（用于生成测试数据）
for i, data in enumerate(train_dataloader):
    logger.info(f\"batch index: {i}, is k_sample_batch: {i in k_samples_idxs}, case index: {data['info_index']}\")
    x = data['x']  # 输入数据（接触地图）

    # 准备模型输入参数
    x_kwargs = {}
    if 'x_mask' in data:
        x_kwargs['x_mask'] = data['x_mask'].to(device)

    # 将所有c_和info_开头的键添加到x_kwargs
    for key in data:
        if key.startswith('c_') or key.startswith('info_'):
            if torch.is_tensor(data[key]):
                x_kwargs[key] = data[key].to(device)
            else:
                x_kwargs[key] = data[key]

    use_k_sample = i in k_samples_idxs
    repeat_times = evaluator.k_samples if use_k_sample else 1

    sample_list_np = []
    k_samples_list_np = []

    # 生成样本（可能需要重复多次）
    for k in range(repeat_times):
        if cfg.model.name.startswith('CMDM'):
            ## 如果使用CMDM模型，输入c_pc_contact包含k个样本
            ## 需要移除这个项，并使用第k个接触地图
            x_kwargs['c_pc_contact'] = data['c_pc_contact'][:, k, :, :].to(device)

        # 使用扩散模型生成样本
        sample = sample_fn(
            model,
            x.shape,
            clip_denoised=False,
            noise=None,
            model_kwargs=x_kwargs,
            progress=True,
        )

        if k == 0:
            for bsi in range(B):
                sample_list_np.append(sample[bsi].cpu().numpy())

        if use_k_sample:
            for bsi in range(B):
                k_samples_list_np.append(sample[bsi].cpu().numpy())

    ## 保存单个样本
    for bsi in range(B):
        res_dict = {'sample': sample_list_np[bsi]}
        for key in data:
            if torch.is_tensor(data[key]):
                res_dict[key] = data[key][bsi].cpu().numpy()
            else:
                res_dict[key] = data[key][bsi]
        sample_list.append(res_dict)

    ## 保存k个样本
    if use_k_sample:
        for bsi in range(B):
            res_dict = {'k_samples': np.stack(k_samples_list_np[bsi::B])}
            for key in data:
                if torch.is_tensor(data[key]):
                    res_dict[key] = data[key][bsi].cpu().numpy()
                else:
                    res_dict[key] = data[key][bsi]
            k_samples_list.append(res_dict)

    ## 如果达到最大样本数，停止生成
    if i + 1 >= evaluator.eval_nbatch:
        break
```

#### 12. 保存生成的map_affordance数据（第287-294行）
```python
## 保存生成的map_affordance数据
final_save_path = os.path.join(train_dir, 'map_affordance_data.npz')
np.savez(final_save_path,
         sample_list=sample_list,
         k_samples_list=k_samples_list,
         total_batches=i+1,
         dataset_size=len(train_dataset))
logger.info(f'map_affordance data saved to {final_save_path}')
```

#### 13. 生成评估报告（第296-298行）
```python
## 生成评估报告
evaluator.evaluate(sample_list, k_samples_list, train_dir, train_dataloader, device=device)
evaluator.report(train_dir)
```

## train_map_affordance 函数与 test 函数的不同

### 1. 数据集来源
| 特性 | test() | train() |
|------|--------|---------|
| **数据集阶段** | `phase='test'` | `phase='train'` |
| **数据集类型** | 测试集 | 训练集 |
| **数据用途** | 评估模型性能 | 生成测试数据集 |

### 2. 输出内容
| 特性 | test() | train() |
|------|--------|---------|
| **主要输出** | 评估指标、测试报告 | 生成的接触地图数据 |
| **保存文件** | 测试结果、评估报告 | `map_affordance_data.npz` |
| **日志文件** | `test.log` | `train.log` |

### 3. 功能目的
| 特性 | test() | train() |
|------|--------|---------|
| **主要目的** | 评估模型性能 | 生成测试数据集 |
| **是否计算指标** | ✅ 是 | ✅ 是 |
| **是否保存数据** | ❌ 否 | ✅ 是 |

### 4. 数据处理
| 特性 | test() | train() |
|------|--------|---------|
| **数据打乱** | `shuffle=False` | `shuffle=False` |
| **批处理大小** | `cfg.task.test.batch_size` | `cfg.task.train.batch_size` |
| **工作进程数** | `cfg.task.test.num_workers` | `cfg.task.train.num_workers` |

### 5. 输出目录
| 特性 | test() | train() |
|------|--------|---------|
| **目录格式** | `eval/test-{timestamp}` | `eval/train-{timestamp}` |
| **主要文件** | `test.log` | `train.log` |
| **数据文件** | ❌ 无 | `map_affordance_data.npz` |

## 使用方法

### 运行测试
```bash
# 默认模式（测试）
python test.py

# 或显式指定
python test.py mode=test
```

### 生成map_affordance测试数据集
```bash
# 生成模式
python test.py mode=train
```

### 使用配置文件
```bash
# 使用特定配置
python test.py task=contact_gen mode=train
```

## 输出文件说明

### test() 输出
```
eval/test-{timestamp}/
├── test.log                      # 运行日志
├── test_results.npz              # 测试结果数据
└── evaluation_report.txt         # 评估报告
```

### train() 输出
```
eval/train-{timestamp}/
├── train.log                     # 运行日志
├── map_affordance_data.npz       # 生成的map_affordance数据
└── evaluation_report.txt         # 评估报告
```

## map_affordance_data.npz 内容

```python
{
    'sample_list': [...],          # 生成的接触地图列表
    'k_samples_list': [...],       # 多样本生成结果
    'total_batches': 100,          # 处理的batch数量
    'dataset_size': 51934          # 数据集大小
}
```

每个样本包含：
- `'sample'`: 生成的接触地图 [N, J]
- `'c_pc_xyz'`: 点云坐标 [N, 3]
- `'c_text'`: 文本描述
- `'info_index'`: 样本索引

## 总结

### test.py 的含义
`test.py` 是 CDM 项目的测试脚本，用于：
1. **评估模型性能**：在测试集上计算各种指标
2. **生成测试数据**：使用训练好的模型生成接触地图

### train_map_affordance 与 test 的不同
1. **数据集**：train使用训练集，test使用测试集
2. **目的**：train生成数据，test评估性能
3. **输出**：train保存生成的数据，test保存评估结果
4. **使用场景**：train用于准备测试数据，test用于验证模型

### 推荐使用
- **模型评估**：使用 `test()` 函数
- **数据生成**：使用 `train()` 函数（mode=train）
- **训练新模型**：使用 `train_contact_gen.py` 脚本
