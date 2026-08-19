import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import sys
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils.flow.x_pred import get_flow_matching_target_velocity  # noqa: E402
from utils.common import instantiate_from_config  # noqa: E402


def load_image(image_path, device, target_size=256):
    image = Image.open(image_path).convert('L')
    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    image = image / 255.0
    image = F.interpolate(image, size=(target_size, target_size),
                          mode='bilinear', align_corners=False)
    return image


def load_model(checkpoint_path, model_type, device):
    cfg = OmegaConf.load("configs/params.yaml")
    if model_type == 'fm':
        model = instantiate_from_config(cfg.model.palette)
    elif model_type == 'gan':
        model = instantiate_from_config(cfg.model.gan.generator)
    elif model_type == 'jit':
        model = instantiate_from_config(cfg.model.jit)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

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


def run_jit_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval, num_ode_steps=50):
    device = lq_batch.device
    N = lq_batch.shape[0]
    gen_sample = torch.randn_like(lq_batch, device=device)
    dt = 1.0 / num_ode_steps
    cond = torch.cat([lq_batch, dpm_batch], dim=1)
    with torch.no_grad():
        for ode_step in range(num_ode_steps):
            t = torch.ones(N, device=device, requires_grad=False) * (ode_step * dt)
            x0_pred_cond = model(gen_sample, t, cond)
            v_t_pred_cond = get_flow_matching_target_velocity(x0_pred_cond, gen_sample, t)
            x0_pred_uncond = model(gen_sample, t, torch.zeros_like(cond, device=device, requires_grad=False))
            v_t_pred_uncond = get_flow_matching_target_velocity(x0_pred_uncond, gen_sample, t)
            v_t_pred = v_t_pred_uncond + cfg_scale_interval * (v_t_pred_cond - v_t_pred_uncond)
            gen_sample = gen_sample + v_t_pred * dt
    return gen_sample


def stitch_patches(patches, n_rows, n_cols):
    """将 n_rows x n_cols 个 patch 按行优先顺序拼回大图"""
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


def get_input_layer_for_group(group_num, group_input_config):
    """
    根据组号查找对应的输入层。
    group_input_config 格式：{(start_group, end_group): input_layer}
    返回 input_layer 或 None（表示该组不需要推理）
    """
    for (start, end), layer in group_input_config.items():
        if start <= group_num <= end:
            return layer
    return None


def process_group_inference(data_root, output_root, layer_min, layer_max,
                             checkpoint_path, group_input_config,
                             model_type='fm', cfg_scale_interval=2,
                             batch_size=8, target_size=256, K=4,
                             n_rows=2, n_cols=2):
    """
    批量处理 data_root 下的所有数字子文件夹，每K个为一组。
    根据 group_input_config 决定每组的输入层，不在配置中的组跳过。
    对于每一组，推理全部子文件夹后按行优先顺序拼接成大图，输出 5 个文件。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 收集并排序所有数字子文件夹
    subfolders = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            try:
                folder_id = int(item)
                subfolders.append((folder_id, item))
            except ValueError:
                pass
    subfolders.sort(key=lambda x: x[0])

    if not subfolders:
        print(f"在 {data_root} 中未找到数字子文件夹")
        return

    model = load_model(checkpoint_path, model_type, device)

    target_layers = list(range(layer_min, layer_max + 1))
    num_layers = len(target_layers)

    # 按 K 个一组分组
    groups = [subfolders[i:i + K] for i in range(0, len(subfolders), K)]

    print(f"共 {len(subfolders)} 个子文件夹，每 {K} 个为一组（{n_rows}x{n_cols}），共 {len(groups)} 组")
    print(f"推理目标层：z{layer_min}~z{layer_max}（共 {num_layers} 层）")
    print(f"组输入配置：{group_input_config}")

    for group_idx, group_folders in enumerate(groups):
        group_num = group_idx + 1
        start_id = group_folders[0][0]
        end_id = group_folders[-1][0]
        group_name = f"{start_id}-{end_id}"

        # 查找该组的输入层
        layer_input = get_input_layer_for_group(group_num, group_input_config)

        print(f"\n{'='*60}")
        print(f"组 {group_num}/{len(groups)}：文件夹 {group_name}")

        if layer_input is None:
            print("  ⏭️  跳过（未在配置中）")
            continue

        print(f"  输入层：z{layer_input}")
        print(f"  目标层：z{layer_min}~z{layer_max}")

        group_output = os.path.join(output_root, f"group{group_num}_{group_name}")
        os.makedirs(group_output, exist_ok=True)

        # dpm 参数（基于该组的输入层）
        dpm_raw = [layer_input - t for t in target_layers]
        dpm_max = max(abs(d) for d in dpm_raw)

        # 收集该组所有 patch 的推理结果
        all_group_preds = []
        all_group_inputs = []
        all_group_gts = []

        for folder_id, folder_name in tqdm(group_folders, desc=f"组{group_num} patch推理", unit="patch", leave=False):
            input_image_path = os.path.join(data_root, folder_name, f"z{layer_input}.png")
            if not os.path.exists(input_image_path):
                print(f"  ⚠️  跳过 {folder_name}：z{layer_input}.png 不存在，用零填充")
                all_group_preds.append(np.zeros((num_layers, target_size, target_size), dtype=np.uint8))
                all_group_inputs.append(np.zeros((target_size, target_size), dtype=np.uint8))
                all_group_gts.append(np.zeros((num_layers, target_size, target_size), dtype=np.uint8))
                continue

            lq = load_image(input_image_path, device, target_size)

            # 推理所有目标层
            dpm_batches = split_into_batches(dpm_raw, batch_size=batch_size)
            all_predictions = []

            for dpm_vals in dpm_batches:
                dpm_normalized = [v / dpm_max for v in dpm_vals]
                lq_batch, dpm_batch = prepare_batch_input(lq, dpm_normalized, 1.0, device=device)
                if model_type == 'fm':
                    output = run_fm_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval)
                elif model_type == 'gan':
                    with torch.no_grad():
                        output = model(lq_batch, dpm_batch)
                elif model_type == 'jit':
                    output = run_jit_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval)
                all_predictions.append(output)

            all_predictions = torch.cat(all_predictions, dim=0)
            pred_np = all_predictions.squeeze(1).cpu().numpy() * 255.0
            pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)
            all_group_preds.append(pred_np)

            # input
            input_save = lq.squeeze().cpu().numpy()
            input_save = np.clip(input_save * 255.0, 0, 255).astype(np.uint8)
            all_group_inputs.append(input_save)

            # GT z-stack
            gt_stack = []
            for z_target in target_layers:
                gt_path = os.path.join(data_root, folder_name, f"z{z_target}.png")
                if os.path.exists(gt_path):
                    frame = Image.open(gt_path).convert('L')
                    if frame.size != (target_size, target_size):
                        frame = frame.resize((target_size, target_size), Image.BILINEAR)
                    gt_stack.append(np.array(frame))
                else:
                    gt_stack.append(np.zeros((target_size, target_size), dtype=np.uint8))
            all_group_gts.append(np.stack(gt_stack, axis=0))

        # 不足 K 个的用零填充补齐
        while len(all_group_preds) < K:
            all_group_preds.append(np.zeros((num_layers, target_size, target_size), dtype=np.uint8))
            all_group_inputs.append(np.zeros((target_size, target_size), dtype=np.uint8))
            all_group_gts.append(np.zeros((num_layers, target_size, target_size), dtype=np.uint8))

        # === 拼接并保存 5 个文件 ===

        # 1. input.png
        stitched_input = stitch_patches(all_group_inputs, n_rows, n_cols)
        Image.fromarray(stitched_input).save(os.path.join(group_output, "input.png"))

        # 2. pred.tif
        pred_tif_frames = []
        for layer_idx in range(num_layers):
            patches = [all_group_preds[p][layer_idx] for p in range(K)]
            stitched = stitch_patches(patches, n_rows, n_cols)
            pred_tif_frames.append(Image.fromarray(stitched))
        pred_tif_path = os.path.join(group_output, "pred.tif")
        pred_tif_frames[0].save(pred_tif_path, save_all=True,
                                append_images=pred_tif_frames[1:], compression="tiff_adobe_deflate")

        # 3. input.tif（GT z-stack）
        input_tif_frames = []
        for layer_idx in range(num_layers):
            patches = [all_group_gts[p][layer_idx] for p in range(K)]
            stitched = stitch_patches(patches, n_rows, n_cols)
            input_tif_frames.append(Image.fromarray(stitched))
        input_tif_path = os.path.join(group_output, "input.tif")
        input_tif_frames[0].save(input_tif_path, save_all=True,
                                 append_images=input_tif_frames[1:], compression="tiff_adobe_deflate")

        # 4 & 5. 随机抽取某一层
        random_z = random.choice(target_layers)
        random_idx = target_layers.index(random_z)

        gt_patches = [all_group_gts[p][random_idx] for p in range(K)]
        stitched_gt = stitch_patches(gt_patches, n_rows, n_cols)
        Image.fromarray(stitched_gt).save(os.path.join(group_output, f"input_z{random_z}.png"))

        pred_patches = [all_group_preds[p][random_idx] for p in range(K)]
        stitched_pred = stitch_patches(pred_patches, n_rows, n_cols)
        Image.fromarray(stitched_pred).save(os.path.join(group_output, f"pred_z{random_z}.png"))

        print(f"  ✅ 组 {group_num} 完成：input.png, input.tif, pred.tif, input_z{random_z}.png, pred_z{random_z}.png")

    print("\n全部处理完成！结果保存在：" + output_root)


if __name__ == "__main__":
    DATA_ROOT = "/data1/azt/cv/recoverZ/data_large_slope"
    data_root = f"{DATA_ROOT}/images"

    model_type = "large_slope"
    output_root = f"outputs/{model_type}/selected"

    layer_min, layer_max = 1, 61       # 推理目标层范围
    K = 4                                # 每 K 个子文件夹为一组
    n_rows = 2                           # 拼接行数
    n_cols = 2                           # 拼接列数
    cfg_scale_interval = 2
    batch_size = 8

    # 组输入配置：(起始组号, 结束组号) → 输入层号
    # 不在此配置中的组将被跳过
    GROUP_INPUT_CONFIG = {
        (1, 5): 61,     # 组 1-5 用第 61 层作为输入
        (8, 8): 61,     # 组 8 用第 61 层作为输入
        (9, 11): 1,     # 组 9-11 用第 1 层作为输入
    }

    network_path = '/data1/azt/cv/recoverZ/outputs/real_slope/fm_palette'
    checkpoint_path = os.path.join(network_path, 'checkpoints/best_val_loss_ema.pt')

    os.makedirs(output_root, exist_ok=True)

    process_group_inference(
        data_root=data_root,
        output_root=output_root,
        layer_min=layer_min,
        layer_max=layer_max,
        checkpoint_path=checkpoint_path,
        group_input_config=GROUP_INPUT_CONFIG,
        model_type='fm',
        cfg_scale_interval=cfg_scale_interval,
        batch_size=batch_size,
        target_size=256,
        K=K,
        n_rows=n_rows,
        n_cols=n_cols,
    )
