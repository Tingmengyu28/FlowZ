# recoverZ

基于深度学习的显微图像 3D 深度重建 (DeepZ)，从单张 2D 显微图像预测完整的 z-stack 三维结构。

## 模型

| 模型 | 描述 | 训练脚本 | 推理脚本 |
|------|------|----------|----------|
| **Flow Matching (FM)** | 基于连续归一化流的深度重建，使用 Palette UNet 架构 | `run/train_fm.py` | `run/inference_fm.py` 等 |
| **GAN** | 生成对抗网络，含 Generator + Discriminator | `run/train_gan.py` | 同上 |
| **JiT** (Joint-in-Time) | Vision Transformer 架构的深度重建 | `run/train_jit.py` | 同上 |
| **SwinIR** | Swin Transformer 图像复原 | — | — |

## 项目结构

```
recoverZ/
├── configs/
│   └── params.yaml              # 模型与训练配置
├── utils/
│   ├── dataset/                 # 数据集加载与变换
│   ├── flow/                    # Flow Matching 前向传播与验证
│   ├── gan/                     # GAN loss 与验证
│   ├── model/                   # 模型定义 (palette, unet, jit, gan, swinir)
│   ├── log/                     # 日志与可视化工具
│   ├── common.py                # 通用工具 (instantiate_from_config)
│   └── ema.py                   # EMA 指数移动平均
├── run/
│   ├── train_fm.py              # FM 训练
│   ├── train_gan.py             # GAN 训练
│   ├── train_jit.py             # JiT 训练
│   ├── inference_fm.py          # 单张推理
│   ├── inference_full_stack.py  # 完整 z-stack 推理
│   ├── inference_brain.py       # 脑组织数据推理 (patch 级别)
│   ├── inference_brain_combine.py # 脑组织拼接推理 (patch → 大图)
│   ├── inference_single_slice.py   # 单 slice 推理
│   ├── inference_multiple_stacks.py # 多 stack 批量推理
│   ├── inference_subfig.py      # 子图推理
│   ├── inference_test.py        # 测试推理
│   ├── scratch_from_std.py      # 从 TIF 提取最清晰层 (基于标准差)
│   ├── scratch_from_maxpool.py  # 从 TIF 提取最清晰层 (基于 maxpool)
│   ├── evaluation_mse.py        # MSE 评估
│   └── evaluation_neighbor.py   # 邻近层评估
├── preworks/                    # 仿真数据预处理
│   ├── gen_simulation_data.py   # 生成仿真数据集
│   ├── gen_psf_from_bbbc006.py  # 从 BBBC006 提取 PSF
│   ├── gen_gt.py                # 生成 ground truth
│   ├── gen_paires.py            # 生成配对数据
│   ├── tif2png.py               # TIF 转 PNG
│   ├── merge_tif.py             # 合并 TIF
│   └── roi_crop.py              # ROI 裁剪
├── preworks_BBBC006/            # BBBC006 数据集预处理
│   ├── tif2png.py
│   ├── gen_pairs.py
│   ├── merge_tif.py
│   └── check_outlier.py
├── preworks_real/               # 真实脑组织数据预处理
│   ├── tif2png_without_folder.py # TIF → PNG 分块 (自定义行列)
│   └── get_pairs.py             # 生成训练/测试配对
├── postprocess/                 # 后处理
│   ├── simulation_get_depth.py  # 仿真深度提取
│   ├── simulation_remove_edge.py # 边缘去除
│   └── wf2con.py                # 宽场转共聚焦
├── configs/
│   └── params.yaml              # 全局配置文件
├── train_fm.sh / train_gan.sh / train_jit.sh  # 训练启动脚本
└── plot.py                      # 绘图工具
```

## 安装

```bash
# 创建 conda 环境
conda create -n recoverz python=3.10 -y
conda activate recoverz

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

### 仿真数据 (BBBC006)

```bash
# 生成仿真数据集
python preworks/gen_simulation_data.py

# TIF 转 PNG
python preworks/tif2png.py
```

### 真实脑组织数据

```bash
# 1. TIF 转 PNG 分块 (可自定义行列)
python preworks_real/tif2png_without_folder.py

# 2. 生成训练/测试配对
python preworks_real/get_pairs.py
```

数据目录结构 (真实数据):
```
data_brain/images/
├── 1_488_Em525_Widefield_/
│   ├── patch_0/
│   │   ├── z1.png
│   │   ├── z2.png
│   │   └── ...
│   ├── patch_1/
│   └── ...
├── 2_488_Em525_Widefield_/
└── ...
```

## 训练

```bash
# Flow Matching 训练
bash train_fm.sh

# 或直接运行
python run/train_fm.py --config configs/params.yaml

# GAN 训练
bash train_gan.sh

# JiT 训练
bash train_jit.sh
```

## 推理

### 单张图像推理

```bash
python run/inference_fm.py
```

### 完整 z-stack 推理

```bash
python run/inference_full_stack.py
```

### 脑组织数据推理 (单 patch)

```bash
python run/inference_brain.py
```

### 脑组织数据推理 (拼接大图)

对 `data_brain/images/{group_name}/` 下所有 patch 进行推理，并按空间布局拼接为完整大图，输出 `input_*.tif` 和 `pred_*.tif`：

```bash
python run/inference_brain_combine.py
```

### 后处理：从 TIF 提取最清晰层

```bash
# 基于标准差
python run/scratch_from_std.py

# 基于 maxpool + gamma 校正
python run/scratch_from_maxpool.py
```

## 配置

主要配置在 `configs/params.yaml`：

- **model**: 模型架构选择 (palette / swinir / unet / jit / gan)
- **dataset**: 数据集配置 (name, channel 数)
- **train**: 训练参数 (batch_size, steps, ema_decay, 各模型输出目录)

## 评估

```bash
# MSE 评估
python run/evaluation_mse.py

# 邻近层评估
python run/evaluation_neighbor.py
```