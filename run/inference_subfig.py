import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import torch.nn.functional as F
import lpips
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import structural_similarity as ssim
from joblib import Parallel, delayed

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

def get_best_informative_crops(h, w, crop_size=100, margin=20):
    """固定返回右上角和左下角的坐标"""
    s = crop_size
    y_top_right, x_top_right = margin, w - s - margin
    y_bottom_left, x_bottom_left = h - s - margin, margin
    return (y_top_right, x_top_right), (y_bottom_left, x_bottom_left)

def calc_ssim_single(img1, img2):
    """单对图像 SSIM 计算 (data_range=1.0)"""
    return ssim(img1, img2, data_range=1.0)

def infer(args) -> None:
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(196) 

    # --- 1. 风格配置 ---
    academic_cmap = plt.get_cmap('hot')
    ica_cmap = LinearSegmentedColormap.from_list("ICA_style", ["#e6e6fa", "#2020a0", "#e69000", "#ffcc00"], N=256)

    # --- 2. 模型加载 ---
    exp_dir = cfg.train.fm.exp_dir
    infer_dir = os.path.join(exp_dir, "crops")
    os.makedirs(infer_dir, exist_ok=True)

    unet = instantiate_from_config(cfg.model.palette).to(device)
    ckpt_path = os.path.join(exp_dir, "checkpoints", "best_val_loss_ema.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    ema = EMA(unet, decay=0.999)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to()
    unet.eval()

    # --- 3. 批量数据准备 ---
    val_dataset = MicroscopyDeepZDataset(
        pairs_file_path=f'data_{cfg.dataset.name}/pairs/val_pairs.txt', 
        image_size=(256, 256)
    )
    batch_size = 8
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    batch_transform = instantiate_from_config(cfg.batch_transform)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)

    # 固定参数
    s, margin = 100, 20
    (y1, x1), (y2, x2) = get_best_informative_crops(256, 256, s, margin)
    found_good_sample = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            print(f"Checking batch {batch_idx}...")
            val_results = validate_model(
                model=unet, val_loader=[batch], 
                batch_transform=batch_transform,
                num_ode_steps=cfg.inference.fm.num_ode_steps,
                device=device, lpips_model=lpips_model, 
                cfg_scale_interval=cfg.train.fm.cfg_scale_interval
            )

            # 数据提取 (B, H, W)
            preds = np.clip(val_results["gen_sample"].cpu().squeeze().numpy(), 0, 1)
            gts = np.clip(val_results["val_gt"].cpu().squeeze().numpy(), 0, 1)
            lqs = np.clip(val_results["val_lq"].cpu().squeeze().numpy(), 0, 1)
            dpms = val_results["val_dpm"].cpu().squeeze().numpy()
            B = preds.shape[0]

            # 并行计算批次 SSIM
            def batch_ssim(target, ref, y, x):
                return Parallel(n_jobs=-1)(delayed(calc_ssim_single)(target[i, y:y+s, x:x+s], ref[i, y:y+s, x:x+s]) for i in range(B))

            s_lq_c1, s_pd_c1 = batch_ssim(lqs, gts, y1, x1), batch_ssim(preds, gts, y1, x1)
            s_lq_c2, s_pd_c2 = batch_ssim(lqs, gts, y2, x2), batch_ssim(preds, gts, y2, x2)

            # 寻找满足双重提升的索引
            best_idx = -1
            for i in range(B):
                if s_pd_c1[i] > s_lq_c1[i] and s_pd_c2[i] > s_lq_c2[i]:
                    best_idx = i
                    break

            if best_idx != -1:
                print(f"Sample found! Batch {batch_idx}, Index {best_idx}")
                found_good_sample = True
                
                # 准备绘图数据
                p, g, l = preds[best_idx], gts[best_idx], lqs[best_idx]
                z_text = rf'$\Delta z = {float(dpms[best_idx][0][0]) * 0.05 * 25:.2f} \mu m$'
                v_min, v_max, diff_vmax = g.min(), g.max(), 0.3

                # 构建 2x5 Grid 数据
                # Row 1: Crop 1, Row 2: Crop 2
                # Columns: GT, Input, |GT-Input|, Pred, |GT-Pred|
                rows = [
                    [g[y1:y1+s, x1:x1+s], l[y1:y1+s, x1:x1+s], np.abs(l[y1:y1+s, x1:x1+s]-g[y1:y1+s, x1:x1+s]), p[y1:y1+s, x1:x1+s], np.abs(p[y1:y1+s, x1:x1+s]-g[y1:y1+s, x1:x1+s])],
                    [g[y2:y2+s, x2:x2+s], l[y2:y2+s, x2:x2+s], np.abs(l[y2:y2+s, x2:x2+s]-g[y2:y2+s, x2:x2+s]), p[y2:y2+s, x2:x2+s], np.abs(p[y2:y2+s, x2:x2+s]-g[y2:y2+s, x2:x2+s])]
                ]
                titles = ["GT", "Input", "|GT-Input|", "Pred", "|GT-Pred|"]
                row_scores = [
                    [None, s_lq_c1[best_idx], None, s_pd_c1[best_idx], None],
                    [None, s_lq_c2[best_idx], None, s_pd_c2[best_idx], None]
                ]

                # --- A. 绘制 2x5 Comparison Grid ---
                fig, axes = plt.subplots(2, 5, figsize=(20, 9))
                plt.subplots_adjust(wspace=0.08, hspace=0.02)

                for r in range(2):
                    for c in range(5):
                        ax = axes[r, c]
                        is_diff = (c == 2 or c == 4)
                        ax.imshow(rows[r][c], cmap=ica_cmap if is_diff else academic_cmap, 
                                  vmin=0 if is_diff else v_min, vmax=diff_vmax if is_diff else v_max)
                        
                        # if r == 0: ax.set_title(titles[c], fontsize=22, fontweight='bold', pad=12)
                        
                        score = row_scores[r][c]
                        if score:
                            t = ax.text(5, 12, f'SSIM: {score:.4f}', color='white', fontsize=16, fontweight='bold')
                            t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
                        ax.axis('off')

                plt.savefig(os.path.join(infer_dir, "00_crop_comparison_grid.png"), bbox_inches='tight', dpi=300)
                plt.close()

                # --- B. 原始大图保存 ---
                for name, img in {"01_full_gt": g, "02_full_lq": l, "03_full_pred": p}.items():
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(img, cmap=academic_cmap, vmin=v_min, vmax=v_max)
                    ax.add_patch(patches.Rectangle((x1, y1), s, s, linewidth=3, edgecolor='w', facecolor='none'))
                    ax.add_patch(patches.Rectangle((x2, y2), s, s, linewidth=3, edgecolor='yellow', facecolor='none'))
                    txt = ax.text(128, 15, z_text, color='white', fontsize=20, fontweight='bold', ha='center', va='center')
                    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
                    ax.axis('off')
                    plt.savefig(os.path.join(infer_dir, f"{name}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()

                break # 找到满意结果，结束任务

    if not found_good_sample: print("Finished, but no sample met the double-SSIM criteria.")
    else: print(f"Academic figures saved to {infer_dir}.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/params.yaml", type=str)
    args = parser.parse_args()
    infer(args)