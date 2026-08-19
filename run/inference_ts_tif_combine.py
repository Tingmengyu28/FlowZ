import os
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from omegaconf import OmegaConf
from PIL import Image
import re
from tqdm import tqdm

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


def ts_inference_combine(group_path, output_root, z_range, checkpoint_path,
                         cfg_scale_interval=2, batch_size=8, target_size=256,
                         t_start=1, t_end=120, tif_z_start=1, tif_z_end=34,
                         n_rows=4, n_cols=4, num_ode_steps=30):
    """
    对时间序列 group 下所有 patch_X 进行推理，按 n_rows x n_cols 空间布局拼接结果。

    输入数据只有时间戳 t（patch_X/t{t}.png），没有 z 层概念。
    每个时间戳 t 作为输入，推理得到一整套输出层 [tif_z_start, tif_z_end]。
    dpm 以目标层范围中点为参考（等价于把时间帧当作中心层），对称分布在 0 附近。

    输出：
      * input.tif: (T, H, W)  原始输入时间序列（拼接后）
      * pred.tif:  (T, C, H, W)  预测结果，T 个时间戳 × C 个输出层（拼接后）

    :param group_path: 如 data_live_cell_0625/images_ts/7-time-channel_1
    :param output_root: 输出目录（结果直接保存到此目录下，不再分子目录）
    :param z_range: dpm 归一化除数
    :param checkpoint_path: 模型 checkpoint 路径
    :param cfg_scale_interval: CFG scale
    :param batch_size: 推理 batch 大小
    :param target_size: 模型输入/输出的空间尺寸
    :param t_start: 时间戳起始号
    :param t_end: 时间戳结束号
    :param tif_z_start: 预测输出层起始号
    :param tif_z_end: 预测输出层结束号
    :param n_rows: 原始分块行数（拼接时反向使用）
    :param n_cols: 原始分块列数
    :param num_ode_steps: Flow Matching ODE 求解步数（默认 30，越小越快，推荐 20~50）
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
    timestamps = list(range(t_start, t_end + 1))
    num_timestamps = len(timestamps)

    # 3. 确定 patch 尺寸（读取第一个 patch 的任一 PNG）
    first_patch = os.path.join(group_path, patch_dirs[0])
    first_t = sorted([f for f in os.listdir(first_patch) if f.endswith('.png')])[0]
    first_img = Image.open(os.path.join(first_patch, first_t)).convert('L')
    patch_h, patch_w = np.array(first_img).shape
    large_h = patch_h * n_rows
    large_w = patch_w * n_cols

    # 4. dpm 参数：以目标层范围中点为参考，对称分布在 0 附近
    #    （时间帧没有 z 属性，等价于将其当作中心层处理）
    center = (tif_z_start + tif_z_end) // 2
    dpm_raw = [t - center for t in tif_layers]
    dpm_max = z_range  # z_range 作为 dpm 归一化除数

    # 每个时间戳的 (patch, layer) 对数 —— 跨 patch 批量推理的总批次
    total_pairs = num_patches * num_layers

    # 内存预估
    pred_bytes = num_timestamps * num_layers * large_h * large_w
    print(f"\n时间戳范围：t{t_start} ~ t{t_end}（共 {num_timestamps} 帧）")
    print(f"推理目标层：z{tif_z_start} ~ z{tif_z_end}（共 {num_layers} 层）")
    print(f"dpm 参考（目标层中点）：{center}，DPM 范围：{min(dpm_raw)} ~ {max(dpm_raw)}，归一化除数：{dpm_max}")
    print(f"ODE 求解步数：{num_ode_steps}，CFG scale：{cfg_scale_interval}")
    print(f"拼接布局：{n_rows}x{n_cols}，Patch 尺寸: {patch_h}x{patch_w}, 大图尺寸: {large_h}x{large_w}")
    print(f"每时间戳批量推理：{num_patches} patch × {num_layers} 层 = {total_pairs} 对，"
          f"batch_size={batch_size}")
    print(f"预计 pred TIF 形状: ({num_timestamps}, {num_layers}, {large_h}, {large_w})  (T, C, H, W)")
    print(f"预计 pred 内存占用: {pred_bytes / 1024 / 1024 / 1024:.2f} GB")

    os.makedirs(output_root, exist_ok=True)

    # 预分配内存
    # input_large_all: (T, H, W) 原始输入拼接
    # pred_large_all:  (T, C, H, W) 预测拼接
    input_large_all = np.zeros((num_timestamps, large_h, large_w), dtype=np.uint8)
    pred_large_all = np.zeros((num_timestamps, num_layers, large_h, large_w), dtype=np.uint8)

    # 预构建 dpm 张量（所有时间戳共用，不随输入图像变化）
    dpm_normalized = [v / dpm_max for v in dpm_raw]
    dpm_single = torch.stack([
        torch.full((1, target_size, target_size), fill_value=v,
                   dtype=torch.float32, device=device)
        for v in dpm_normalized
    ], dim=0)  # (num_layers, 1, H, W)
    # 复制到所有 patch：(num_patches * num_layers, 1, H, W)，排列 [p0_l0..p0_lN, p1_l0.., ...]
    dpm_batch_full = dpm_single.repeat(num_patches, 1, 1, 1)

    # 5. 逐时间戳推理（跨 patch 批量推理，充分利用 GPU）
    num_chunks_per_ts = (total_pairs + batch_size - 1) // batch_size
    total_steps = num_timestamps * num_chunks_per_ts
    pbar = tqdm(total=total_steps, desc="推理进度", unit="batch")

    for ti, t_val in enumerate(timestamps):
        # 5.1 加载该时间戳所有 patch 的输入图像
        lq_list = []
        valid = True
        for patch_dir_name in patch_dirs:
            lq_path = os.path.join(group_path, patch_dir_name, f"t{t_val}.png")
            if not os.path.exists(lq_path):
                print(f"\n  错误：{lq_path} 不存在，跳过 t={t_val}")
                valid = False
                break
            lq_list.append(load_image(lq_path, device, target_size))

        if not valid:
            continue

        # 5.2 构建跨 patch 的批量输入：(num_patches * num_layers, 1, H, W)
        #     每个 patch 的图像重复 num_layers 次，与 dpm_batch_full 排列对齐
        lq_batch_full = torch.cat([lq.repeat(num_layers, 1, 1, 1) for lq in lq_list], dim=0)

        # 5.3 分 chunk 推理（batch_size 现在真正控制单次 forward 的样本数）
        all_outputs = []
        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)
            output = run_fm_inference_batch(
                model, lq_batch_full[start:end], dpm_batch_full[start:end],
                cfg_scale_interval, num_ode_steps=num_ode_steps)
            all_outputs.append(output)
            pbar.update(1)
            pbar.set_postfix({"t": f"{t_val}/{timestamps[-1]}",
                              "chunk": f"{start // batch_size + 1}/{num_chunks_per_ts}"})

        # 5.4 合并并 reshape: (total_pairs, 1, H, W) → (num_patches, num_layers, H, W)
        all_outputs = torch.cat(all_outputs, dim=0).view(
            num_patches, num_layers, target_size, target_size)
        # 一次性搬到 CPU（比逐 patch 逐层搬运快得多）
        all_preds_np = (all_outputs.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # 5.5 拼接 input（按时间戳）
        all_patch_inputs = []
        for lq in lq_list:
            input_save = lq.squeeze().cpu().numpy()
            input_save = np.clip(input_save * 255.0, 0, 255).astype(np.uint8)
            input_img = Image.fromarray(input_save).resize((patch_w, patch_h), Image.BILINEAR)
            all_patch_inputs.append(np.array(input_img))
        stitched_input = stitch_patches(all_patch_inputs, n_rows, n_cols)
        input_large_all[ti] = stitched_input

        # 5.6 拼接 pred（按层）
        for layer_idx in range(num_layers):
            pred_patches = []
            for pi in range(num_patches):
                pred_img = Image.fromarray(all_preds_np[pi, layer_idx]).resize(
                    (patch_w, patch_h), Image.BILINEAR)
                pred_patches.append(np.array(pred_img))
            stitched_pred = stitch_patches(pred_patches, n_rows, n_cols)
            pred_large_all[ti, layer_idx] = stitched_pred

    pbar.close()

    # 8. 保存 input.tif  (T, H, W) —— 用 PIL 多页 TIF
    input_tif_path = os.path.join(output_root, f"input_t{t_start}_t{t_end}.tif")
    input_frames = [Image.fromarray(input_large_all[ti]) for ti in range(num_timestamps)]
    input_frames[0].save(input_tif_path, save_all=True,
                         append_images=input_frames[1:], compression="tiff_adobe_deflate")
    print(f"\nInput TIF 已保存：{input_tif_path}  ({num_timestamps} 帧, {large_w}x{large_h})")

    # 9. 保存 pred.tif  (T, C, H, W) —— 用 tifffile 保存 4D TIF
    pred_tif_path = os.path.join(output_root,
                                 f"pred_t{t_start}_t{t_end}_z{tif_z_start}_z{tif_z_end}.tif")
    try:
        tifffile.imwrite(pred_tif_path, pred_large_all,
                         bigtiff=True, compression='zlib',
                         metadata={'axes': 'TCYX'})
        print(f"Pred TIF 已保存：{pred_tif_path}  "
              f"({num_timestamps}x{num_layers} 帧, {large_w}x{large_h}, shape={pred_large_all.shape})")
    except Exception as e:
        print(f"⚠️  4D TIF 保存失败 ({e})，改为按时间戳分文件保存")
        for ti in range(num_timestamps):
            t_pred_path = os.path.join(output_root, f"pred_t{timestamps[ti]}.tif")
            tifffile.imwrite(t_pred_path, pred_large_all[ti],
                             bigtiff=True, compression='zlib',
                             metadata={'axes': 'CYX'})
        print(f"  已保存 {num_timestamps} 个独立 pred_t*.tif 文件")

    # 10. 保存首帧 input 和 首层 pred 预览 PNG
    if num_timestamps > 0:
        preview_path = os.path.join(output_root, f"input_t{timestamps[0]}.png")
        Image.fromarray(input_large_all[0]).save(preview_path)
        print(f"Input 首帧预览已保存：{preview_path}")

    if num_timestamps > 0 and num_layers > 0:
        pred_preview_path = os.path.join(output_root,
                                         f"pred_t{timestamps[0]}_z{tif_z_start}.png")
        Image.fromarray(pred_large_all[0, 0]).save(pred_preview_path)
        print(f"Pred 首帧首层预览已保存：{pred_preview_path}")

    print(f"\n全部完成，结果保存在：{output_root}")


if __name__ == "__main__":
    DATA_ROOT = "live_cell_0625"
    image_id = "7-time-channel_2"
    group_path = f"data_{DATA_ROOT}/images_ts/{image_id}"
    output_root = f"outputs/{DATA_ROOT}/ts_combined/{image_id}"
    z_range = 20
    cfg_scale_interval = 2
    batch_size = 512            # 一次 forward 处理 512 张图（336对只需1个batch）
    target_size = 128
    num_ode_steps = 20            # ODE 步数（15=极快，20=快，30=默认，50=高质量）
    t_start, t_end = 1, 60          # 时间戳范围
    tif_z_start, tif_z_end = 1, 21   # 预测输出层范围
    n_rows = 4
    n_cols = 4

    # network_path = f'/data1/azt/cv/recoverZ/outputs/{DATA_ROOT}/fm_palette'
    network_path = '/data1/azt/cv/recoverZ/outputs/live_cell/fm_palette'
    checkpoint_path = os.path.join(network_path, 'checkpoints/best_val_loss_ema.pt')

    os.makedirs(output_root, exist_ok=True)

    ts_inference_combine(
        group_path=group_path,
        output_root=output_root,
        z_range=z_range,
        checkpoint_path=checkpoint_path,
        cfg_scale_interval=cfg_scale_interval,
        batch_size=batch_size,
        target_size=target_size,
        t_start=t_start,
        t_end=t_end,
        tif_z_start=tif_z_start,
        tif_z_end=tif_z_end,
        n_rows=n_rows,
        n_cols=n_cols,
        num_ode_steps=num_ode_steps,
    )
