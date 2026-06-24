import os
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim

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


def compute_clarity(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def brain_inference(data_root, output_root, z, z_range, checkpoint_path,
                    cfg_scale_interval=2, batch_size=8, target_size=256,
                    tif_z_start=1, tif_z_end=34):
    """
    对给定的 data_root 进行推理：
      - 读取 data_root/z{z}.png 作为输入
      - 对 tif_z_start ~ tif_z_end 所有层做推理（z_range 仅用于 dpm 归一化）
      - 选出结构最清晰的5层保存到 output_root
      - 输出 input 和 pred 的 TIF
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_path = os.path.join(data_root, f"z{z}.png")
    if not os.path.exists(input_path):
        print(f"输入图像不存在：{input_path}")
        return

    model = load_model(checkpoint_path, device)

    # 推理范围由 tif_z_start/tif_z_end 控制，z_range 仅用于 dpm 归一化
    target_layers = list(range(tif_z_start, tif_z_end + 1))
    dpm_raw = [t - z for t in target_layers]
    dpm_max = z_range

    print(f"输入层：z{z}，目标层：z{tif_z_start}~z{tif_z_end}")
    print(f"DPM 范围：{min(dpm_raw)}~{max(dpm_raw)}，归一化除数：{dpm_max}")

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

    # 保存输入图像（原始 + 归一化）
    input_np = lq.squeeze().cpu().numpy() * 255.0
    input_raw = np.clip(input_np, 0, 255).astype(np.uint8)
    input_norm = ((input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(input_raw).save(os.path.join(output_root, "input_raw.png"))
    Image.fromarray(input_norm).save(os.path.join(output_root, "input_norm.png"))

    # 加载 target（z10），固定作为 SSIM 参考
    target_path = os.path.join(data_root, "z10.png")
    if os.path.exists(target_path):
        target_img = load_image(target_path, device, target_size)
        target_np = target_img.squeeze().cpu().numpy() * 255.0
        target_raw = np.clip(target_np, 0, 255).astype(np.uint8)
        target_norm = ((target_np - target_np.min()) / (target_np.max() - target_np.min() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(target_raw).save(os.path.join(output_root, "target_raw.png"))
        Image.fromarray(target_norm).save(os.path.join(output_root, "target_norm.png"))
        use_target = True
    else:
        print(f"target z10 不存在：{target_path}，回退为 input 自身")
        target_norm = input_norm
        use_target = False

    all_results = []
    for i, z_target in enumerate(target_layers):
        pred = all_predictions[i].squeeze().cpu().numpy()
        pred_raw = (pred * 255.0)
        p_min, p_max = pred_raw.min(), pred_raw.max()
        pred_raw = np.clip(pred_raw, 0, 255).astype(np.uint8)
        pred_norm = ((pred_raw.astype(np.float32) - p_min) / (p_max - p_min + 1e-8) * 255).astype(np.uint8)
        clarity = compute_clarity(pred_norm)
        ssim_val = ssim(target_norm, pred_norm, data_range=255)
        all_results.append((z_target, clarity, ssim_val, pred_raw, pred_norm))

    if not all_results:
        print("没有可处理的图像")
        return

    os.makedirs(output_root, exist_ok=True)

    def save_group(results, group_name, output_root):
        group_dir = os.path.join(output_root, group_name)
        os.makedirs(group_dir, exist_ok=True)
        print(f"\n=== {group_name} ===")
        for rank, (zt, clarity, ssim_val, pred_raw, pred_norm) in enumerate(results):
            cv2.putText(pred_norm, f"SSIM:{ssim_val:.4f} C:{clarity:.0f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
            Image.fromarray(pred_norm).save(os.path.join(group_dir, f"top{rank + 1}_z{zt}.png"))
            print(f"  #{rank + 1}: z{zt}, clarity={clarity:.2f}, SSIM={ssim_val:.4f}")

    # Group 1: 按清晰度排序的 Top 5
    top5_clarity = sorted(all_results, key=lambda x: x[1], reverse=True)[:5]
    save_group(top5_clarity, "top5_clarity", output_root)

    # Group 2: 归一化后按 SSIM 排序的 Top 5
    top5_ssim = sorted(all_results, key=lambda x: x[2], reverse=True)[:5]
    save_group(top5_ssim, "top5_ssim", output_root)

    # 保存边界层（z_min 和 z_max），仅在推理范围内时保存
    z_min = z - z_range
    z_max = z + z_range
    for boundary_z in [z_min, z_max]:
        if boundary_z < tif_z_start or boundary_z > tif_z_end:
            print(f"边界层 z{boundary_z} 不在推理范围内，跳过")
            continue
        for zt, clarity, ssim_val, pred_raw, pred_norm in all_results:
            if zt == boundary_z:
                # 原始（未归一化）
                raw_save_path = os.path.join(output_root, f"boundary_z{boundary_z}_raw.png")
                Image.fromarray(pred_raw).save(raw_save_path)
                # 归一化
                norm_save_path = os.path.join(output_root, f"boundary_z{boundary_z}.png")
                label = f"z{boundary_z}  SSIM:{ssim_val:.4f}  C:{clarity:.0f}"
                cv2.putText(pred_norm.copy(), label, (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
                Image.fromarray(pred_norm).save(norm_save_path)
                print(f"边界层已保存：{raw_save_path} (raw), {norm_save_path} (norm)")
                break

    print(f"\n全部完成，结果保存在：{output_root}")

    # === 导出 TIF（直接复用已有推理结果，无需二次推理）===
    tif_predictions_np = all_predictions.squeeze(1).cpu().numpy() * 255.0
    p_min, p_max = tif_predictions_np.min(), tif_predictions_np.max()
    tif_frames = []
    for i in range(len(target_layers)):
        pred_norm = ((tif_predictions_np[i] - p_min) / (p_max - p_min + 1e-8) * 255).astype(np.uint8)
        tif_frames.append(Image.fromarray(pred_norm))

    pred_tif_path = os.path.join(output_root, f"pred_z{tif_z_start}_z{tif_z_end}.tif")
    tif_frames[0].save(pred_tif_path, save_all=True, append_images=tif_frames[1:], compression="tiff_adobe_deflate")
    print(f"Pred TIFF 已保存：{pred_tif_path}")

    # 收集 data_root 中所有 z*.png，按 z 层号筛选并构建 input TIF
    def extract_z(fname):
        return int(''.join(filter(str.isdigit, fname.replace('.png', ''))))

    input_z_files = sorted(
        [f for f in os.listdir(data_root) if f.endswith('.png')],
        key=extract_z
    )
    if input_z_files:
        h_out, w_out = tif_frames[0].size[::-1]  # (H, W)
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
            print(f"Input TIFF 已保存：{input_tif_path}  ({len(input_frames)} 帧, {w_out}x{h_out})")
        else:
            print(f"在 {data_root} 中未找到 z{tif_z_start}~z{tif_z_end} 对应的 PNG")

    print(f"\n全部完成，结果保存在：{output_root}")


if __name__ == "__main__":
    data_type = "BBBC006"
    image_idx = 's2/p17'
    data_root = f"/data1/azt/cv/recoverZ/data_{data_type}/images/{image_idx}"
    output_root = f"outputs/{data_type}/selected/{image_idx}"
    z = 30
    z_range = 15
    cfg_scale_interval = 2
    batch_size = 8
    tif_z_start, tif_z_end = 1, 34

    network_path = f'/data1/azt/cv/recoverZ/outputs/{data_type}/fm_palette'
    checkpoint_path = os.path.join(network_path, 'checkpoints/best_val_loss_ema.pt')

    os.makedirs(output_root, exist_ok=True)

    brain_inference(
        data_root=data_root,
        output_root=output_root,
        z=z,
        z_range=z_range,
        checkpoint_path=checkpoint_path,
        cfg_scale_interval=cfg_scale_interval,
        batch_size=batch_size,
        target_size=256,
        tif_z_start=tif_z_start,
        tif_z_end=tif_z_end,
    )