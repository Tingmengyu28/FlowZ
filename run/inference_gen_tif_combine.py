import os
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import re

import sys
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils.common import instantiate_from_config  # noqa: E402


def load_image(image_path, device, target_size=256):
    """读取单张 PNG，resize 到 target_size，返回 (1,1,H,W) 归一化 tensor"""
    image = Image.open(image_path).convert('L')
    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    image = image / 255.0
    image = F.interpolate(image, size=(target_size, target_size),
                          mode='bilinear', align_corners=False)
    return image


def load_model(checkpoint_path, device):
    cfg = OmegaConf.load("configs/params.yaml")
    model = instantiate_from_config(cfg.model.palette)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def prepare_batch_input(lq, dpm_batch_values, L_max, device):
    batch_size = len(dpm_batch_values)
    H, W = lq.shape[2], lq.shape[3]
    lq_batch = lq.repeat(batch_size, 1, 1, 1)
    dpm_batch = []
    for value in dpm_batch_values:
        value = value / L_max
        dpm_single = torch.full((1, H, W), fill_value=value,
                                dtype=torch.float32, device=device)
        dpm_batch.append(dpm_single)
    return lq_batch, torch.stack(dpm_batch, dim=0)


def run_fm_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval, num_ode_steps=50):
    device = lq_batch.device
    N = lq_batch.shape[0]
    gen_sample = torch.randn_like(lq_batch, device=device)
    dt = 1.0 / num_ode_steps
    cond = torch.cat([lq_batch, dpm_batch], dim=1)
    with torch.no_grad():
        for ode_step in range(num_ode_steps):
            t_current = torch.ones(N, device=device) * (ode_step * dt)
            v_cond = model(gen_sample, t_current, cond)
            v_uncond = model(gen_sample, t_current,
                             torch.zeros_like(cond, device=device, requires_grad=False))
            v_final = cfg_scale_interval * v_cond + (1 - cfg_scale_interval) * v_uncond
            gen_sample = gen_sample + v_final * dt
    return gen_sample


def split_into_batches(full_list, batch_size=8):
    batches = []
    for i in range(0, len(full_list), batch_size):
        batches.append(full_list[i:i + batch_size])
    return batches


def extract_patch_index(patch_dir_name):
    """从 patch_X 中提取数字 X"""
    match = re.search(r'patch_(\d+)', patch_dir_name)
    return int(match.group(1)) if match else -1


def extract_z_index(filename):
    """从 zX.png 中提取数字 X"""
    match = re.search(r'z(\d+)', filename)
    return int(match.group(1)) if match else -1


def stitch_patches(patches, n_rows, n_cols):
    """
    将 n_rows x n_cols 个 patch 按行优先拼回大图。
    patches: 按行优先排列的 (H_patch, W_patch) numpy 数组列表
    返回: (n_rows*H_patch, n_cols*W_patch) 大图
    """
    patch_h, patch_w = patches[0].shape
    large_h = patch_h * n_rows
    large_w = patch_w * n_cols
    large = np.zeros((large_h, large_w), dtype=patches[0].dtype)
    for idx, patch in enumerate(patches):
        row = idx // n_cols
        col = idx % n_cols
        large[row * patch_h:(row + 1) * patch_h,
              col * patch_w:(col + 1) * patch_w] = patch
    return large


def brain_inference_combine(group_path, output_root, z, z_range, checkpoint_path,
                            cfg_scale_interval=2, batch_size=8, target_size=256,
                            tif_z_start=1, tif_z_end=34, n_rows=4, n_cols=4):
    """
    对 group 下所有 patch_X 进行推理，按 n_rows x n_cols 空间布局拼接结果。

    流程：
      1. 扫描 group_path/patch_X 目录，按 patch 索引排序
      2. 对每个 patch 读取 z{z}.png 作为输入，推理全部目标层 [tif_z_start, tif_z_end]
      3. 按 z 层收集所有 patch 的 pred，拼接成大图
      4. 按 z 层收集所有 patch 的 input 图像，拼接成大图
      5. 输出 input_*.tif 和 pred_*.tif

    :param group_path: 如 data_brain/images/4_488_Em525_Widefield_
    :param output_root: 输出目录
    :param z: 输入 z 层号（每个 patch 读取 z{z}.png）
    :param z_range: dpm 归一化除数（控制模型对距离的敏感度）
    :param checkpoint_path: 模型 checkpoint 路径
    :param cfg_scale_interval: CFG scale
    :param batch_size: 推理 batch 大小
    :param target_size: 模型输入/输出的空间尺寸
    :param tif_z_start: TIF 起始 z 层
    :param tif_z_end: TIF 结束 z 层
    :param n_rows: 原始分块行数（拼接时反向使用）
    :param n_cols: 原始分块列数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 收集所有 patch 目录，按索引排序
    patch_dirs = sorted(
        [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d)) and d.startswith('patch_')],
        key=extract_patch_index
    )
    if not patch_dirs:
        print(f"错误：{group_path} 下未找到 patch_X 子目录")
        return
    num_patches = len(patch_dirs)
    if num_patches != n_rows * n_cols:
        print(f"警告：patch 数量 ({num_patches}) 与 n_rows*n_cols ({n_rows}x{n_cols}={n_rows*n_cols}) 不匹配")

    print(f"找到 {num_patches} 个 patch: {patch_dirs}")

    # 2. 加载模型
    model = load_model(checkpoint_path, device)

    tif_layers = list(range(tif_z_start, tif_z_end + 1))
    num_layers = len(tif_layers)

    # dpm 参数：输入层 z 到目标层 t 的距离
    dpm_raw = [t - z for t in tif_layers]
    dpm_max = z_range  # z_range 作为 dpm 归一化除数

    print(f"输入层：z{z}")
    print(f"推理目标层：z{tif_z_start} ~ z{tif_z_end}（共 {num_layers} 层）")
    print(f"DPM 范围：{min(dpm_raw)} ~ {max(dpm_raw)}，归一化除数：{dpm_max}")
    print(f"拼接布局：{n_rows}x{n_cols}")

    os.makedirs(output_root, exist_ok=True)

    # 3. 确定 patch 尺寸（读取第一个 patch 的任一 PNG）
    first_patch = os.path.join(group_path, patch_dirs[0])
    first_z = sorted([f for f in os.listdir(first_patch) if f.endswith('.png')])[0]
    first_img = Image.open(os.path.join(first_patch, first_z)).convert('L')
    patch_h, patch_w = np.array(first_img).shape
    large_h = patch_h * n_rows
    large_w = patch_w * n_cols
    print(f"Patch 尺寸: {patch_h}x{patch_w}, 大图尺寸: {large_h}x{large_w}")

    # 4. 对每个 patch 进行推理
    all_patch_preds = []  # [num_patches, num_layers, H_pred, W_pred]

    for pi, patch_dir_name in enumerate(patch_dirs):
        patch_path = os.path.join(group_path, patch_dir_name)
        lq_path = os.path.join(patch_path, f"z{z}.png")
        if not os.path.exists(lq_path):
            print(f"错误：{lq_path} 不存在，中止")
            return

        lq = load_image(lq_path, device, target_size)

        # 批处理该 patch 的所有目标层
        dpm_batches = split_into_batches(dpm_raw, batch_size=batch_size)
        patch_pred_list = []
        for dpm_vals in dpm_batches:
            dpm_normalized = [v / dpm_max for v in dpm_vals]
            lq_batch, dpm_batch = prepare_batch_input(lq, dpm_normalized, 1.0, device=device)
            output = run_fm_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval)
            patch_pred_list.append(output)

        patch_pred = torch.cat(patch_pred_list, dim=0)  # (num_layers, 1, H, W)
        all_patch_preds.append(patch_pred)
        print(f"  Patch [{pi + 1}/{num_patches}] {patch_dir_name} 推理完成")

    # 5. 按 z 层拼接 input 和 pred
    input_frames = []
    pred_frames = []

    for layer_idx, target_z in enumerate(tif_layers):
        # --- 拼接 input ---
        input_patches = []
        for patch_dir_name in patch_dirs:
            patch_path = os.path.join(group_path, patch_dir_name)
            z_path = os.path.join(patch_path, f"z{target_z}.png")
            if os.path.exists(z_path):
                img = Image.open(z_path).convert('L')
                patch_np = np.array(img)
                if patch_np.shape != (patch_h, patch_w):
                    img = img.resize((patch_w, patch_h), Image.BILINEAR)
                    patch_np = np.array(img)
                input_patches.append(patch_np)
            else:
                # 某些 z 层可能不存在，用零填充
                input_patches.append(np.zeros((patch_h, patch_w), dtype=np.uint8))
        large_input = stitch_patches(input_patches, n_rows, n_cols)
        input_frames.append(Image.fromarray(large_input))

        # --- 拼接 pred ---
        pred_patches = []
        for pi in range(num_patches):
            pred_tensor = all_patch_preds[pi][layer_idx]  # (1, 1, H, W)
            pred_np = pred_tensor.squeeze().cpu().numpy() * 255.0
            pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)
            # resize 回 patch 尺寸（模型输出 target_size x target_size）
            pred_img = Image.fromarray(pred_np).resize((patch_w, patch_h), Image.BILINEAR)
            pred_patches.append(np.array(pred_img))
        large_pred = stitch_patches(pred_patches, n_rows, n_cols)
        pred_frames.append(Image.fromarray(large_pred))

        if (layer_idx + 1) % 10 == 0 or layer_idx == 0:
            print(f"  拼接进度: {layer_idx + 1}/{num_layers}")

    # 6. 保存 input TIF
    input_tif_path = os.path.join(output_root, f"input_z{tif_z_start}_z{tif_z_end}.tif")
    input_frames[0].save(input_tif_path, save_all=True,
                         append_images=input_frames[1:], compression="tiff_adobe_deflate")
    print(f"\nInput TIF 已保存：{input_tif_path}  ({len(input_frames)} 帧, {large_w}x{large_h})")

    # 7. 保存 pred TIF
    pred_tif_path = os.path.join(output_root, f"pred_z{tif_z_start}_z{tif_z_end}.tif")
    pred_frames[0].save(pred_tif_path, save_all=True,
                        append_images=pred_frames[1:], compression="tiff_adobe_deflate")
    print(f"Pred TIF 已保存：{pred_tif_path}  ({len(pred_frames)} 帧, {large_w}x{large_h})")

    # 8. 从 input TIF 中提取第 z 层保存为 PNG
    z_layer_idx = z - tif_z_start
    if 0 <= z_layer_idx < len(input_frames):
        z_png_path = os.path.join(output_root, f"input_z{z}.png")
        input_frames[z_layer_idx].save(z_png_path)
        print(f"Input z{z} 层已保存：{z_png_path}")
    else:
        print(f"警告：z{z} 不在 TIF 范围 [{tif_z_start}, {tif_z_end}] 内，无法提取")

    print(f"\n全部完成，结果保存在：{output_root}")


if __name__ == "__main__":
    DATA_ROOT = "live_cell"
    image_id = "plane 6 z-stack"
    group_path = f"data_{DATA_ROOT}/images/{image_id}"
    output_root = f"outputs/{DATA_ROOT}/combined/{image_id}"
    z = 15
    z_range = 20
    cfg_scale_interval = 2
    batch_size = 8
    target_size = 256
    tif_z_start, tif_z_end = 1, 41
    n_rows = 4
    n_cols = 4

    network_path = f'/data1/azt/cv/recoverZ/outputs/{DATA_ROOT}/fm_palette'
    checkpoint_path = os.path.join(network_path, 'checkpoints/best_val_loss_ema.pt')

    os.makedirs(output_root, exist_ok=True)

    brain_inference_combine(
        group_path=group_path,
        output_root=output_root,
        z=z,
        z_range=z_range,
        checkpoint_path=checkpoint_path,
        cfg_scale_interval=cfg_scale_interval,
        batch_size=batch_size,
        target_size=target_size,
        tif_z_start=tif_z_start,
        tif_z_end=tif_z_end,
        n_rows=n_rows,
        n_cols=n_cols,
    )