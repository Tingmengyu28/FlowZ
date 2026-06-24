import os
import warnings
from argparse import ArgumentParser
from omegaconf import OmegaConf
from matplotlib.colors import LinearSegmentedColormap
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import lpips
import sys

# 基础路径配置
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils.dataset.dataset import MicroscopyDeepZDataset 
from utils.common import instantiate_from_config 
from utils.flow.v_pred import validate_model 
from utils.ema import EMA 

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def set_seed(seed, device_specific=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device_specific and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def infer(args) -> None:
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(231, device_specific=True)
    
    exp_dir = cfg.train.fm.exp_dir
    infer_dir = os.path.join(exp_dir, "inference_results")
    os.makedirs(infer_dir, exist_ok=True)

    # 模型加载逻辑
    unet = instantiate_from_config(cfg.model.palette)
    unet.to(device)
    ema_decay = getattr(cfg.train.fm, "ema_decay", cfg.train.ema_decay)
    ema = EMA(unet, decay=ema_decay)
    ckpt_path = os.path.join(exp_dir, "checkpoints", "best_val_loss_ema.pt")
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to()
        print(f"Loaded best EMA model from: {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    unet.eval()

    # 数据集配置
    if cfg.dataset.ch >= 0:
        if cfg.dataset.name == 'BBBC006':
            val_dataset = MicroscopyDeepZDataset(pairs_file_path='data_BBBC006/pairs/test_pairs.txt', image_size=(256, 256)) 
        elif cfg.dataset.name == 'exp':
            val_dataset = MicroscopyDeepZDataset(pairs_file_path=f'data/pairs/ch{cfg.dataset.ch}/val_pairs.txt', image_size=(256, 256))
        elif cfg.dataset.name == 'simulation':
            val_dataset = MicroscopyDeepZDataset(pairs_file_path='data_simulation/pairs/val_pairs.txt', image_size=(256, 256))
        elif cfg.dataset.name == 'real_plain':
            val_dataset = MicroscopyDeepZDataset(pairs_file_path='data_real_plain/pairs/val_pairs.txt', image_size=(256, 256))
        elif cfg.dataset.name == 'real_slope':
            val_dataset = MicroscopyDeepZDataset(pairs_file_path='data_real_slope/pairs/val_pairs.txt', image_size=(256, 256))
    else:
        raise ValueError("Invalid dataset channel configuration")

    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.inference.batch_size, shuffle=False)
    batch_transform = instantiate_from_config(cfg.batch_transform)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpips_model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)

    # 执行推理
    with torch.no_grad():
        val_results = validate_model(
            model=unet, val_loader=val_loader, batch_transform=batch_transform,
            num_ode_steps=cfg.inference.fm.num_ode_steps, device=device,
            lpips_model=lpips_model, cfg_scale_interval=cfg.train.fm.cfg_scale_interval,
            top_n=128
        )

    # 数据准备
    gen_sample = val_results["gen_sample"].cpu()
    val_gt = val_results["val_gt"].cpu()
    val_lq = val_results["val_lq"].cpu()
    val_dpm = val_results["val_dpm"].cpu()
    all_val_loss = torch.tensor(val_results["val_losses"]).cpu()

    # 筛选 DPM > 0.5 的样本
    filtered_indices = [i for i in range(gen_sample.shape[0]) if val_dpm[i][0,0,0].item() > 0.5]
    filtered_samples = []
    for idx in filtered_indices:
        filtered_samples.append({
            "loss": all_val_loss[idx // cfg.inference.batch_size], # 简化处理 loss 对应
            "lq": val_lq[idx], "gt": val_gt[idx], "pred": gen_sample[idx]
        })

    filtered_samples.sort(key=lambda x: x["loss"])
    k = min(8, len(filtered_samples))
    top_samples = filtered_samples[:k]

    # 提取并计算差异
    top_lq = torch.stack([s["lq"] for s in top_samples])
    top_gt = torch.stack([s["gt"] for s in top_samples])
    top_pred = torch.stack([s["pred"] for s in top_samples])
    diff_input_pred = torch.abs(top_lq - top_pred)
    diff_gt_pred = torch.abs(top_gt - top_pred)

    # --- 核心绘图改进部分 (ICA 风格 & 降低对比度) ---
    row_titles = ['Input', 'GT', 'Pred', '|Input-Pred|', '|GT-Pred|']
    data_rows = [top_lq, top_gt, top_pred, diff_input_pred, diff_gt_pred]
    
    # 1. 定义 Colormaps
    # ICA 风格：淡紫 -> 深蓝 -> 橙色 -> 亮黄
    ica_colors = ["#dadaf0", "#202080", "#e68a00", "#ffcc00"]
    ica_cmap = LinearSegmentedColormap.from_list("ICA_style", ica_colors, N=256)
    
    # Fire 风格：使用 inferno 或 hot (热力图感)
    # 如果环境安装了 colorcet，可以用 colorcet.cm.fire
    fire_cmap = plt.get_cmap('inferno') 

    # 2. 动态对比度调整 (保持你之前的降低对比度逻辑)
    vmax_img = torch.quantile(top_pred.flatten(), 0.999).item() * 1.2
    vmax_diff = torch.quantile(diff_input_pred.flatten(), 0.995).item() * 1.2

    fig, axes = plt.subplots(5, k, figsize=(k * 3, 13), gridspec_kw={'hspace': 0.08, 'wspace': 0.02})
    
    im_fire = None
    im_ica = None

    for r in range(5):
        row_data = data_rows[r].squeeze(1).numpy()
        for c in range(k):
            ax = axes[r, c]
            if r < 3:
                # 前三行：使用 Fire 风格 (Input, GT, Pred)
                im_fire = ax.imshow(row_data[c], cmap=fire_cmap, vmin=0, vmax=vmax_img)
            else:
                # 后二行：使用 ICA 风格 (Error Maps)
                im_ica = ax.imshow(row_data[c], cmap=ica_cmap, vmin=0, vmax=vmax_diff)
            
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                # 遵循指令：确保标题使用 GT 缩写
                ax.set_ylabel(row_titles[r], fontsize=30, fontweight='bold', rotation=0, ha='right', va='center')

    # 3. 添加双 Colorbar 并调整位置
    # 上方 Colorbar 对应 Fire
    cax1 = fig.add_axes([0.93, 0.48, 0.012, 0.35]) 
    fig.colorbar(im_fire, cax=cax1, label='Intensity (Fire Style)')
    
    # 下方 Colorbar 对应 ICA
    cax2 = fig.add_axes([0.93, 0.15, 0.012, 0.22])
    fig.colorbar(im_ica, cax=cax2, label='Error (ICA Style)')

    # 保存路径
    grid_path = os.path.join(infer_dir, "ICA_style_results.png")
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {grid_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/params.yaml", type=str)
    args = parser.parse_args()
    infer(args)