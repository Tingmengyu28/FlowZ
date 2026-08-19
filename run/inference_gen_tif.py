import os
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import sys
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils.common import instantiate_from_config


def load_image(image_path, device, target_size=256):
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


def brain_inference(data_root, output_root, z, z_range, target_index, checkpoint_path,
                    cfg_scale_interval=2, batch_size=8, target_size=256,
                    tif_z_start=1, tif_z_end=34):
    """
    对给定的 data_root 进行推理：
      - 读取 data_root/z{z}.png 作为输入
      - 对 tif_z_start ~ tif_z_end 所有层做推理（z_range 仅用于 dpm 归一化）
      - 保存：input PNG、input TIF、pred TIF、target_gt PNG、target_pred PNG
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_path = os.path.join(data_root, f"z{z}.png")
    if not os.path.exists(input_path):
        print(f"输入图像不存在：{input_path}")
        return

    model = load_model(checkpoint_path, device)

    target_layers = list(range(tif_z_start, tif_z_end + 1))
    dpm_raw = [t - z for t in target_layers]
    dpm_max = z_range

    print(f"输入层：z{z}，目标层：z{tif_z_start}~z{tif_z_end}")
    print(f"DPM 范围：{min(dpm_raw)}~{max(dpm_raw)}，归一化除数：{dpm_max}")
    print(f"Target 层：z{target_index}")

    lq = load_image(input_path, device, target_size)
    dpm_batches = split_into_batches(dpm_raw, batch_size=batch_size)
    all_predictions = []

    for dpm_vals in dpm_batches:
        dpm_normalized = [v / dpm_max for v in dpm_vals]
        lq_batch, dpm_batch = prepare_batch_input(lq, dpm_normalized, 1.0, device=device)
        output = run_fm_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval)
        all_predictions.append(output)

    all_predictions = torch.cat(all_predictions, dim=0)
    os.makedirs(output_root, exist_ok=True)

    # 1. 保存 input PNG（raw + norm）
    input_np = lq.squeeze().cpu().numpy() * 255.0
    input_raw = np.clip(input_np, 0, 255).astype(np.uint8)
    input_norm = ((input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(input_raw).save(os.path.join(output_root, f"input_z{z}_raw.png"))
    Image.fromarray(input_norm).save(os.path.join(output_root, f"input_z{z}_norm.png"))
    print(f"Input PNG 已保存：input_z{z}_raw.png, input_z{z}_norm.png")

    # 2. 保存 pred TIF
    tif_predictions_np = all_predictions.squeeze(1).cpu().numpy() * 255.0
    p_min, p_max = tif_predictions_np.min(), tif_predictions_np.max()
    tif_frames = []
    for i in range(len(target_layers)):
        pred_norm = ((tif_predictions_np[i] - p_min) / (p_max - p_min + 1e-8) * 255).astype(np.uint8)
        tif_frames.append(Image.fromarray(pred_norm))

    pred_tif_path = os.path.join(output_root, f"pred_z{tif_z_start}_z{tif_z_end}.tif")
    tif_frames[0].save(pred_tif_path, save_all=True, append_images=tif_frames[1:], compression="tiff_adobe_deflate")
    print(f"Pred TIF 已保存：{pred_tif_path}")

    # 3. 保存 input TIF（从 data_root 收集 GT PNG）
    def extract_z(fname):
        return int(''.join(filter(str.isdigit, fname.replace('.png', ''))))

    h_out, w_out = tif_frames[0].size[::-1]
    input_z_files = sorted(
        [f for f in os.listdir(data_root) if f.endswith('.png')],
        key=extract_z
    )
    input_frames = []
    for fname in input_z_files:
        z_val = extract_z(fname)
        if z_val < tif_z_start or z_val > tif_z_end:
            continue
        frame = Image.open(os.path.join(data_root, fname)).convert('L')
        if frame.size != (w_out, h_out):
            frame = frame.resize((w_out, h_out), Image.BILINEAR)
        input_frames.append(frame)

    if input_frames:
        input_tif_path = os.path.join(output_root, f"input_z{tif_z_start}_z{tif_z_end}.tif")
        input_frames[0].save(input_tif_path, save_all=True,
                             append_images=input_frames[1:], compression="tiff_adobe_deflate")
        print(f"Input TIF 已保存：{input_tif_path}  ({len(input_frames)} 帧)")
    else:
        print(f"在 {data_root} 中未找到 z{tif_z_start}~z{tif_z_end} 对应的 PNG")

    # 4. 保存 target_gt PNG（GT 的 target_index 层，raw + norm）
    target_gt_path = os.path.join(data_root, f"z{target_index}.png")
    if os.path.exists(target_gt_path):
        gt_img = Image.open(target_gt_path).convert('L')
        if gt_img.size != (w_out, h_out):
            gt_img = gt_img.resize((w_out, h_out), Image.BILINEAR)
        gt_np = np.array(gt_img).astype(np.float32)
        gt_norm = ((gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(np.clip(gt_np, 0, 255).astype(np.uint8)).save(
            os.path.join(output_root, f"target_gt_z{target_index}_raw.png"))
        Image.fromarray(gt_norm).save(
            os.path.join(output_root, f"target_gt_z{target_index}_norm.png"))
        print(f"Target GT PNG 已保存：target_gt_z{target_index}_raw.png, target_gt_z{target_index}_norm.png")
    else:
        print(f"Target GT 不存在：{target_gt_path}")

    # 5. 保存 target_pred PNG（pred 的 target_index 层，raw + norm）
    if target_index < tif_z_start or target_index > tif_z_end:
        print(f"Target index {target_index} 不在推理范围 [{tif_z_start}, {tif_z_end}] 内")
    else:
        target_idx_in_tif = target_index - tif_z_start
        pred_raw_np = tif_predictions_np[target_idx_in_tif]
        pred_raw_uint8 = np.clip(pred_raw_np, 0, 255).astype(np.uint8)
        pred_norm = ((pred_raw_np - pred_raw_np.min()) / (pred_raw_np.max() - pred_raw_np.min() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(pred_raw_uint8).save(
            os.path.join(output_root, f"target_pred_z{target_index}_raw.png"))
        Image.fromarray(pred_norm).save(
            os.path.join(output_root, f"target_pred_z{target_index}_norm.png"))
        print(f"Target Pred PNG 已保存：target_pred_z{target_index}_raw.png, target_pred_z{target_index}_norm.png")

    print(f"\n全部完成，结果保存在：{output_root}")


if __name__ == "__main__":
    data_type = "BBBC006"
    image_idx = 's1/k15'
    data_root = f"/data1/azt/cv/recoverZ/data_{data_type}/images/{image_idx}"
    output_root = f"outputs/{data_type}/selected/{image_idx}"
    z = 28
    z_range = 15
    target_index = 18
    cfg_scale_interval = 2
    batch_size = 8
    tif_z_start, tif_z_end = 13, 34

    network_path = f'/data1/azt/cv/recoverZ/outputs/{data_type}/fm_palette'
    checkpoint_path = os.path.join(network_path, 'checkpoints/best_val_loss_ema.pt')

    os.makedirs(output_root, exist_ok=True)

    brain_inference(
        data_root=data_root,
        output_root=output_root,
        z=z,
        z_range=z_range,
        target_index=target_index,
        checkpoint_path=checkpoint_path,
        cfg_scale_interval=cfg_scale_interval,
        batch_size=batch_size,
        target_size=256,
        tif_z_start=tif_z_start,
        tif_z_end=tif_z_end,
    )