# recoverZ

基于深度学习的显微图像 3D 深度重建 (DeepZ)，从单张 2D 显微图像预测完整的 z-stack 三维结构。

## 工作流程

```
原始 TIF/仿真数据  ──→  preworks（预处理）──→  train_pairs.txt / val_pairs.txt
                                                          │
                                                          ▼
                                                 修改 configs/params.yaml
                                                          │
                                                          ▼
                                              run/train_fm.py  （训练）
                                                          │
                                                          ▼
                                              run/inference_*.py  （推理）
```

## 1. 预处理 (preworks) — 三选一

### 方式 A: 仿真数据

```bash
python preworks/gen_simulation_data.py          # 生成仿真数据集
python preworks/tif2png.py                       # TIF 转 PNG
python preworks/gen_paires.py                     # 生成训练/测试配对
```

### 方式 B: BBBC006 公开数据

```bash
python preworks_BBBC006/tif2png.py              # TIF 转 PNG
python preworks_BBBC006/gen_pairs.py             # 生成训练/测试配对
python preworks_BBBC006/merge_tif.py             # 合并 TIF（可选）
```

### 方式 C: 真实脑组织数据

```bash
# 1. TIF → PNG 分块（可自定义 N_ROWS/N_COLS）
python preworks_real_images/tif2png_without_folder.py

# 2. 生成训练/测试配对（支持 group 排除）
python preworks_real_images/get_pairs.py
```

预处理完成后，会在 `data_{dataset_name}/pairs/` 下生成 `train_pairs.txt` 和 `val_pairs.txt`。

## 2. 配置

修改 `configs/params.yaml`：

```yaml
dataset:
  name: brain                    # 数据集名，对应 data_{name}/
  ch: -1                         # channel 号（-1 表示无 channel 区分）

model:
  palette:                       # Flow Matching 模型
  jit:                           # JiT 模型
  gan:                           # GAN 模型

train:
  batch_size: 32
  train_steps: 240000
  fm:
    exp_dir: outputs/{name}/fm_palette    # FM 输出目录
    learning_rate: 1e-4
  gan:
    exp_dir: outputs/{name}/gan_beyond    # GAN 输出目录
  jit:
    exp_dir: outputs/{name}/jit           # JiT 输出目录
```

关键参数说明：

| 参数 | 说明 |
|------|------|
| `dataset.name` | 数据集名称，训练时会从 `data_{name}/pairs/` 读取配对文件 |
| `model.palette` | Flow Matching 的 UNet 架构 |
| `train.fm.exp_dir` | FM 模型保存路径 |
| `train.fm.learning_rate` | FM 学习率 |
| `train.batch_size` | 训练 batch size |
| `train.train_steps` | 总训练步数 |

## 3. 训练

三种模型可选：

```bash
# Flow Matching（推荐）
python run/train_fm.py --config configs/params.yaml
bash train_fm.sh

# GAN
python run/train_gan.py --config configs/params.yaml
bash train_gan.sh

# JiT (Joint-in-Time)
python run/train_jit.py --config configs/params.yaml
bash train_jit.sh
```

训练完成后，checkpoint 保存在 `outputs/{dataset_name}/{model}/checkpoints/` 下，最佳模型为 `best_val_loss_ema.pt`。

## 4. 推理

```bash
# 单张图像推理
python run/inference_fm.py

# 生成 TIF 堆栈（多 z 层推理）
python run/inference_gen_tif.py

# 生成拼接 TIF（patch → 大图）
python run/inference_gen_tif_combine.py

# 测试推理
python run/inference_test.py
```

各推理脚本可在 `__main__` 中调整参数（data_root、z、z_range、checkpoint_path 等）。

## 项目结构

```
recoverZ/
├── configs/params.yaml              # 全局配置
├── utils/                           # 核心工具库
│   ├── model/                       # 模型定义 (palette, unet, jit, gan, swinir)
│   ├── dataset/                     # 数据集加载
│   ├── flow/                        # Flow Matching 前向传播
│   ├── gan/                         # GAN loss
│   ├── log/                         # 日志与可视化
│   ├── common.py                    # instantiate_from_config
│   └── ema.py                       # EMA
├── run/
│   ├── train_fm.py / train_gan.py / train_jit.py  # 训练
│   ├── inference_fm.py              # 单张推理
│   ├── inference_gen_tif.py         # TIF 堆栈生成
│   ├── inference_gen_tif_combine.py # 拼接大图 TIF
│   └── inference_test.py            # 测试推理
├── preworks/                        # 仿真数据预处理
│   ├── gen_simulation_data.py       # 生成仿真数据
│   ├── tif2png.py                   # TIF 转 PNG
│   ├── gen_paires.py                # 生成配对
│   └── ...
├── preworks_BBBC006/                # BBBC006 数据预处理
│   ├── tif2png.py
│   ├── gen_pairs.py
│   └── merge_tif.py
├── preworks_real_images/            # 真实脑组织数据预处理
│   ├── tif2png_without_folder.py    # TIF → 分块 PNG
│   ├── tif2png_with_folder.py       # TIF → PNG（保留目录）
│   └── get_pairs.py                 # 生成配对（支持 group 排除）
├── postprocess/                     # 后处理工具
├── train_fm.sh / train_gan.sh / train_jit.sh  # 训练脚本
└── plot.py                          # 绘图
```

## 数据目录结构

```
data_{name}/
├── images/                          # PNG 图像
│   ├── group_1/
│   │   ├── patch_0/
│   │   │   ├── z1.png
│   │   │   ├── z2.png
│   │   │   └── ...
│   │   ├── patch_1/
│   │   └── ...
│   └── group_2/
└── pairs/                           # 配对文件
    ├── train_pairs.txt
    └── val_pairs.txt
```

## 安装

```bash
conda create -n recoverz python=3.10 -y
conda activate recoverz
pip install -r requirements.txt
```