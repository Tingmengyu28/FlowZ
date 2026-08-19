# recoverZ

基于深度学习的显微图像 3D 深度重建（DeepZ）项目。核心思想是：**输入单张 2D 显微图像 + 一个深度偏移量 `dpm`（defocus propagation model 参数），用 Flow Matching 模型预测该深度处的切片图像**，从而从一张或多张 2D 图重建出完整的 z-stack（或时间序列）三维结构。

主要特点：

- **Flow Matching（FM）扩散模型**：训练稳定、采样质量高，是当前主推模型。
- **dpm 条件注入**：将「目标层相对输入层的深度距离」归一化后作为空间条件，与输入图拼接送入 UNet。
- **CFG（Classifier-Free Guidance）**：训练时随机丢弃条件，推理时用条件/无条件外推增强质量。
- **多数据源支持**：仿真数据、BBBC006 公开数据、真实显微数据（z-stack 与时间序列两种）。
- **大图分块重建**：先转换成 `patch_x` 分块推理，再按 `n_rows × n_cols` 拼回大图。

---

## 目录结构与整体流程

```
原始 TIF / 仿真数据
      │
      ▼  ① 数据处理（preworks / preworks_real_images）
   PNG 分块（data_*/images, images_ts）
      │
      ▼  ② 生成配对（get_pairs.py）
   train_pairs.txt / val_pairs.txt（data_*/pairs/）
      │
      ▼  ③ 配置（configs/params.yaml）
      │
      ▼  ④ 训练（bash train_fm.sh）
   outputs/{dataset.name}/fm_palette/checkpoints/best_val_loss_ema.pt
      │
      ▼  ⑤ 推理（run/inference_*.py）
   input/pred 的 TIF 与 PNG
```

---

## 1. 环境搭建

### 1.1 创建环境

推荐使用 Conda 新建独立环境（Python 3.10）：

```bash
conda create -n recoverz python=3.10 -y
conda activate recoverz
```

### 1.2 安装依赖

项目根目录的 `requirements.txt` 是一份**完整的 `pip freeze` 导出**，包含大量与本项目无关的包（ROS2、TensorFlow、nnUNet 等）。若只是跑本项目，**建议手动安装核心依赖**：

```bash
pip install torch torchvision            # 按平台选择对应 CUDA 版本
pip install accelerate omegaconf         # 分布式训练 + 配置解析
pip install numpy scipy scikit-image     # 数值与图像处理
pip install opencv-python pillow         # 图像读写
pip install tifffile                     # TIF 读写（含 4D T,C,H,W）
pip install tqdm matplotlib              # 进度条与绘图
pip install lpips tensorboard            # 损失与训练可视化
pip install einops                       # 张量重排
```

> 也可以直接 `pip install -r requirements.txt`，但会拉取大量无关依赖，速度慢且可能产生版本冲突。

### 1.3 硬件与分布式

- 训练使用 `accelerate` 启动多卡分布式训练，混合精度 `fp16`。
- 项目在 NVIDIA L40（46GB）上验证过，单卡即可推理，训练默认使用 4 卡。

---

## 2. 数据处理

数据处理的最终目标：把原始 TIF 转换成「`group / patch_x / z*.png`」或「`group / patch_x / t*.png`」的分块 PNG 结构，再生成训练/验证配对文件。

### 2.1 数据格式约定（重要）

训练数据目录结构（以真实 z-stack 数据为例）：

```
data_{dataset_name}/
├── tif/                          # 原始 TIF（可选，仅作备份）
└── images/                       # 转换后的 PNG
    └── {图像ID}/                 # 即 group（通常是 TIF 文件名去掉扩展名）
        └── patch_0/
            ├── z1.png
            ├── z2.png
            └── ...
        └── patch_1/
            └── ...
```

时间序列数据使用独立的 `images_ts/` 目录，切片命名为 `t1.png, t2.png, ...`：

```
data_{dataset_name}/
└── images_ts/
    └── {图像ID}/
        └── patch_0/
            ├── t1.png
            ├── t2.png
            └── ...
```

### 2.2 方式一：真实显微数据（当前主用）

脚本位于 `preworks_real_images/`：

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `tif2png_without_folder.py` | 批量处理整个 TIF 目录（z-stack） | `data_*/tif/*.tif` | `data_*/images/{id}/patch_x/z*.png` |
| `tif2png_with_folder.py` | 批量处理，保留子目录层级 | TIF 目录 | 保留层级的分块 PNG |
| `tif2png_single_tif.py` | 处理单个 z-stack TIF | 单个 TIF 路径 | `output_root/{id}/patch_x/z*.png` |
| `tif2png_ts_single_tif.py` | 处理单个时间序列 TIF（按时间戳保存） | 单个时间序列 TIF | `output_root/{id}/patch_x/t*.png` |

**z-stack 转换示例**（修改 `tif2png_without_folder.py` 的 `main()` 内参数）：

```python
DATA_ROOT = "data_live_cell_0625"
INPUT_ROOT = f"{DATA_ROOT}/tif"
OUTPUT_ROOT = f"{DATA_ROOT}/images"
N_ROWS = 4          # 行方向切块数
N_COLS = 4          # 列方向切块数
SLICE_STEP = 1      # 每隔多少层保存一层
```

**时间序列转换示例**（修改 `tif2png_ts_single_tif.py` 的 `main()`）：

```python
DATA_ROOT = "data_live_cell_0625"
TIFF_PATH = f"{DATA_ROOT}/tif/.../7-time-channel_1.tif"
OUTPUT_ROOT = f"{DATA_ROOT}/images_ts"
```

时间序列 TIF 通常是 `(T, H, W)` 或 `(T, C, H, W)` 维度，脚本会自动探测并逐时间戳保存为 `t{t}.png`（多通道时加 `_c{c}` 后缀）。

> 归一化方式：所有切片/时间帧**共享同一个全局最大值**做 min-max 缩放到 `[0, 255]`，保证后续分块拼接无缝。

### 2.3 方式二：仿真数据

脚本位于 `preworks/`，生成仿真点光源 z-stack：

```bash
python preworks/gen_psf_from_bbbc006.py    # 先生成 PSF（需要 data_simulation/PSF_bbbc006.tif）
python preworks/gen_simulation_data.py     # 生成仿真 z-stack（400 组 × 21 层，256×256）
python preworks/tif2png.py                 # TIF → PNG
python preworks/gen_paires.py              # 生成配对（老版，按 channel 区分）
```

### 2.4 方式三：BBBC006 公开数据

脚本位于 `preworks_BBBC006/`：

```bash
python preworks_BBBC006/tif2png.py         # TIF → PNG
python preworks_BBBC006/gen_pairs.py       # 生成配对
python preworks_BBBC006/merge_tif.py       # 合并 TIF（可选）
```

### 2.5 生成配对文件（get_pairs.py）

`preworks_real_images/get_pairs.py` 是当前主用的配对生成脚本。它会：

1. 自动探测目录结构（`group/patch_x/zX.png` 或 `group/zX.png`）。
2. 以 **group 粒度**划分训练/测试集（可指定 `EXCLUDE_GROUPS` 强制进入测试集）。
3. 在每个 patch 内部生成「输入层 → 目标层」配对，`dpm = (目标层 z - 输入层 z) / L_MAX` 归一化。

关键配置（`main()` 内）：

```python
DATA_ROOT = "/data1/azt/cv/recoverZ/data_live_cell"
INPUT_DIR = f"{DATA_ROOT}/images"          # 所有 group 所在目录
OUTPUT_DIR = f"{DATA_ROOT}/pairs"          # 配对输出目录
L_MIN = 1                                  # 最小 z 差值（绝对值）
L_MAX = 20                                 # 最大 z 差值（绝对值，也是归一化除数）
TRAIN_RATIO = 0.9                          # 训练占比
TRAIN_MAX_PAIRS = 150000                   # 训练集上限
TEST_MAX_PAIRS = 5000                      # 测试集上限
EXCLUDE_GROUPS = ["plane 6 z-stack"]       # 强制进入测试集的 group（子串匹配）
```

运行后生成：

```
data_{dataset_name}/pairs/
├── train_pairs.txt    # 每行: 输入图路径 \t 目标图路径 \t 归一化dpm
└── val_pairs.txt
```

> **dpm 归一化约定**：训练配对里的 `dpm = z_diff / L_MAX`，推理时也必须用同样的除数 `z_range = L_MAX`（见推理章节），两者必须一致，否则模型对深度的敏感度会偏移。

---

## 3. 配置（configs/params.yaml）

核心配置：

```yaml
dataset:
  name: live_cell       # 数据集名，训练时读取 data_{name}/pairs/
  ch: 2                 # channel 号（>=0 时走 data_{name}/pairs/ 分支）

model:
  palette:              # Flow Matching 使用的 UNet（in_channel=3: 1 输入图 + 2 dpm 拼接通道取 2）
    target: utils.model.palette.UNet
    params:
      image_size: 256
      in_channel: 3
      out_channel: 1
      inner_channel: 32
      channel_mults: [1, 2, 4, 8, 8]
      attn_res: [8]

train:
  batch_size: 32
  num_workers: 16
  train_steps: 240000
  gradient_accumulation_steps: 4
  val_every: 1000
  ema_decay: 0.999
  fm:
    exp_dir: /data1/azt/cv/recoverZ/outputs/${dataset.name}/fm_palette
    learning_rate: 1e-4
    label_drop_prob: 0.1      # CFG 条件丢弃概率
    cfg_scale_interval: 2
    Beta_alpha: 2.0
    Beta_beta: 2.0

inference:
  fm:
    num_ode_steps: 100        # 训练验证阶段使用的 ODE 步数
```

关键参数：

| 参数 | 说明 |
|------|------|
| `dataset.name` | 数据集名，训练会从 `data_{name}/pairs/train_pairs.txt` 读取 |
| `dataset.ch` | channel 号；`>=0` 时读取 `data_{name}/pairs/`（无 channel 子目录） |
| `model.palette` | FM 使用的 UNet 架构，`in_channel=3`（1 张灰度输入 + 2 通道 dpm 条件） |
| `train.exp_dir` / `train.fm.exp_dir` | 模型与 checkpoint 输出目录 |
| `train.fm.label_drop_prob` | CFG 的标签丢弃概率 |
| `inference.fm.num_ode_steps` | 训练期验证用的 ODE 采样步数 |

---

## 4. 训练

### 4.1 启动训练（Flow Matching，推荐）

用 `accelerate` 多卡分布式训练。入口脚本 `train_fm.sh`：

```bash
bash train_fm.sh
```

脚本内容（关键点）：

```bash
export PYTHONPATH="/data1/azt/cv/recoverZ:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,4   # 使用的 GPU

accelerate launch \
    --main_process_port=29501 \
    --num_processes=4 \
    --mixed_precision=fp16 \
    run/train_fm.py \
    --config configs/params.yaml
```

也可以单卡直接运行：

```bash
python run/train_fm.py --config configs/params.yaml
```

### 4.2 其它模型（GAN / JiT）

```bash
bash train_gan.sh     # 对应 run/train_gan.py
bash train_jit.sh     # 对应 run/train_jit.py
```

> 三个训练脚本均使用 `configs/params.yaml` 配置。

### 4.3 训练输出

训练完成后，模型保存在：

```
outputs/{dataset.name}/fm_palette/
├── checkpoints/
│   ├── best_val_loss_ema.pt    # 最佳验证损失对应的 EMA 模型（推理使用这个）
│   └── final_ema_model.pt      # 训练结束时的 EMA 模型
└── events.out.tfevents.*       # TensorBoard 日志
```

> 推理统一加载 `checkpoints/best_val_loss_ema.pt`，其内部包含 `ema` 权重。

---

## 5. 推理

推理脚本均位于 `run/`，参数在各自 `__main__` 中配置。

| 脚本 | 用途 |
|------|------|
| `inference_fm.py` | 基于配对文件的单张/批量推理（验证集风格），保存推理结果 |
| `inference_gen_tif.py` | 单组数据生成 z-stack TIF |
| `inference_gen_tif_combine.py` | z-stack 大图拼接推理（patch → 大图，支持多输入 z） |
| `inference_ts_tif_combine.py` | 时间序列大图拼接推理（多时间戳，输出 T,C,H,W） |
| `inference_large_slope.py` | 按组配置不同输入层的大规模推理 |
| `inference_test.py` | 早期单张推理/测试脚本 |

### 5.1 z-stack 拼接推理（inference_gen_tif_combine.py）

```python
# __main__ 关键参数
DATA_ROOT = "live_cell_0625"
image_id = "7-Z_stack-channel_1"
group_path = f"data_{DATA_ROOT}/images/{image_id}"      # 输入 patch 目录
output_root = f"outputs/{DATA_ROOT}/combined/{image_id}"
z = [16, 19, 22]            # 输入 z 层（int 或 list[int]）
z_range = 20                # dpm 归一化除数，必须等于 get_pairs 的 L_MAX
cfg_scale_interval = 2
batch_size = 8
tif_z_start, tif_z_end = 1, 41     # 目标层范围
n_rows, n_cols = 4, 4              # 原始分块布局
checkpoint_path = ".../outputs/live_cell/fm_palette/checkpoints/best_val_loss_ema.pt"
```

运行后，每个输入 z 生成一个独立子目录：

```
output_root/
├── input_z16/
│   ├── input_z1_z41.tif     # 拼接后的输入 z-stack
│   ├── pred_z1_z41.tif      # 拼接后的预测 z-stack
│   └── input_z16.png        # 输入层 z16 的 PNG
├── input_z19/ ...
└── input_z22/ ...
```

### 5.2 时间序列拼接推理（inference_ts_tif_combine.py）

输入只有时间戳 `t1.png, t2.png, ...`，没有 z 层概念；每个时间戳作为输入，推理一整套目标 z 层：

```python
# __main__ 关键参数
image_id = "7-time-channel_1"
group_path = f"data_{DATA_ROOT}/images_ts/{image_id}"
output_root = f"outputs/{DATA_ROOT}/ts_combined/{image_id}"
z_range = 20                # dpm 归一化除数
batch_size = 512            # 单次 forward 处理的 image 数
num_ode_steps = 20          # ODE 采样步数（越小越快，20~50）
t_start, t_end = 1, 60          # 时间戳范围
tif_z_start, tif_z_end = 1, 21  # 每个时间戳推理的输出层范围
n_rows, n_cols = 4, 4
```

输出：

```
output_root/
├── input_t1_t60.tif                # (T, H, W) 原始输入时间序列
├── pred_t1_t60_z1_z21.tif          # (T, C, H, W) 预测结果（4D）
├── input_t1.png                    # 首帧预览
└── pred_t1_z1.png                  # 首帧首层预览
```

**推理性能调优建议**：

- `batch_size` 控制单次 forward 的样本数；在显存允许范围内尽量调大（如 256~512）。
- 真正的速度瓶颈是 **ODE 步数** `num_ode_steps`（每步 = 2 次 CFG forward，一次条件一次无条件）。降低步数可线性加速，FM 模型通常 20 步即可有不错质量。
- 换用更大的 `n_rows/n_cols` 前，确认与 `tif2png` 时的分块布局一致。

---

## 6. 评估

- `run/evaluation_mse.py`：计算 GT 与预测深度图的 MSE。
- `run/evaluation_neighbor.py`：相邻层一致性评估。
- `get_quantitative_results.py`：批量量化结果汇总。
- `plot.py`：结果可视化绘图。

---

## 7. 项目结构

```
recoverZ/
├── configs/params.yaml              # 全局配置（模型/数据集/训练/推理）
├── utils/                           # 核心工具库
│   ├── model/                       # 模型定义（palette/unet/jit/gan/swinir 等）
│   ├── dataset/                     # 数据集与配对加载（MicroscopyDeepZDataset）
│   ├── flow/                        # Flow Matching 前向/速度预测
│   ├── gan/                         # GAN loss / 验证
│   ├── log/                         # 日志与图像可视化
│   ├── common.py                    # instantiate_from_config 等
│   └── ema.py                       # EMA
├── run/
│   ├── train_fm.py / train_gan.py / train_jit.py   # 训练入口
│   ├── inference_fm.py / inference_gen_tif.py       # 推理
│   ├── inference_gen_tif_combine.py                # z-stack 拼接推理
│   ├── inference_ts_tif_combine.py                 # 时间序列拼接推理
│   ├── inference_large_slope.py / inference_test.py
│   └── evaluation_mse.py / evaluation_neighbor.py  # 评估
├── preworks/                        # 仿真数据预处理
│   ├── gen_simulation_data.py / gen_psf_from_bbbc006.py
│   ├── tif2png.py / merge_tif.py / gen_paires.py
│   └── roi_crop.py / gen_gt.py
├── preworks_BBBC006/                # BBBC006 预处理
│   ├── tif2png.py / gen_pairs.py / merge_tif.py / check_outlier.py
├── preworks_real_images/            # 真实数据预处理
│   ├── tif2png_without_folder.py / tif2png_with_folder.py
│   ├── tif2png_single_tif.py / tif2png_ts_single_tif.py
│   └── get_pairs.py
├── postprocess/                     # 后处理（深度提取/边缘去除等）
├── train_fm.sh / train_gan.sh / train_jit.sh      # 训练脚本
├── requirements.txt
└── README.md
```

---

## 8. 常见问题


- **推理结果偏暗/偏亮**：检查推理脚本的 `z_range` 是否与 `get_pairs.py` 的 `L_MAX` 一致（dpm 归一化除数必须匹配）。
- **拼接出现缝隙或重叠**：确认推理脚本的 `n_rows/n_cols` 与 `tif2png` 时的分块布局一致。
- **时间序列 TIF 维度**：`tif2png_ts_single_tif.py` 会打印输入 shape；`(T,H,W)` 视作单通道时间序列，`(T,C,H,W)` 会按通道分别保存。
- **显存不足**：降低 `batch_size` 或 `num_ode_steps`；4D 预测 TIF（T,C,H,W）会占用较大内存，必要时减小 `t_end` 或 `tif_z_end`。