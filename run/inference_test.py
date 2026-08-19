import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim

import sys
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils.flow.x_pred import get_flow_matching_target_velocity  # noqa: E402
from utils.common import instantiate_from_config  # noqa: E402


def load_image(image_path, device, target_size=256):
    """
    加载并预处理图像，返回作为条件的低质量图像（核心新增：缩放图像到target_size x target_size）
    """
    image = Image.open(image_path).convert('L')  # 转换为灰度图像
    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    image = image / 255.0  # 归一化
    
    image = F.interpolate(
        image,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )
    
    return image

def load_model(checkpoint_path, model_type, device):
    """加载模型和检查点"""
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
    """
    准备单个批次的输入（dpm张量自动跟随lq的256x256尺寸）
    """
    batch_size = len(dpm_batch_values)
    H, W = lq.shape[2], lq.shape[3]  # 此时H=256，W=256（跟随缩放后的图像）
    lq_batch = lq.repeat(batch_size, 1, 1, 1)  # (batch_size, 1, 256, 256)
    
    # 构建dpm批次 (batch_size, 1, 256, 256)，每个样本的HxW所有位置均为对应dpm值
    dpm_batch = []
    for value in dpm_batch_values:
        value = value / L_max
        dpm_single = torch.full((1, H, W), fill_value=value, dtype=torch.float32, device=device)
        dpm_batch.append(dpm_single)
    dpm_batch = torch.stack(dpm_batch, dim=0)  # (batch_size, 1, 256, 256)
    
    return lq_batch, dpm_batch

def run_fm_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval, num_ode_steps=50):
    """使用Flow Matching求解进行批量推理"""
    device = lq_batch.device
    N = lq_batch.shape[0]
    
    gen_sample = torch.randn_like(lq_batch, device=device)
    dt = 1.0 / num_ode_steps
    cond = torch.cat([lq_batch, dpm_batch], dim=1)
    
    with torch.no_grad():
        for ode_step in range(num_ode_steps):
            t_current = torch.ones(N, device=device) * (ode_step * dt)
            v_cond = model(gen_sample, t_current, cond)
            v_uncond = model(gen_sample, t_current, torch.zeros_like(cond, device=device, requires_grad=False))
            v_final = cfg_scale_interval * v_cond + (1 - cfg_scale_interval) * v_uncond

            gen_sample = gen_sample + v_final * dt
    
    return gen_sample

def run_jit_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval, num_ode_steps=50):
    """使用JIT编译后的Flow Matching模型进行批量推理"""
    device = lq_batch.device
    N = lq_batch.shape[0]
    
    # 初始化为随机噪声
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

def save_original_images(output, dpm_batch_values, original_subdir):
    """保存原始推理图像到output_dir下的新建子文件夹"""
    # 确保子文件夹存在
    os.makedirs(original_subdir, exist_ok=True)
    for i, value in enumerate(dpm_batch_values):
        result = output[i].squeeze().cpu().numpy()
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)  # 反归一化并裁剪到有效范围
        result = Image.fromarray(result)
        result.save(os.path.join(original_subdir, f'z50_dpm_{value}.png'))

def channel_wise_max_pooling(all_40_images, output_dir):
    """
    将40张图像作为通道，执行通道维度的Max Pooling，得到一张结果图并保存到output_dir根目录
    :param all_40_images: 40张图像的张量，形状为(40, 1, 256, 256)
    """
    all_40_images = all_40_images.squeeze(dim=1)  # 去除多余的单通道，变为(40, 256, 256)
    all_40_images = all_40_images.unsqueeze(dim=0)  # 增加批次维度，变为(1, 40, 256, 256)
    
    max_pooled_result, _ = torch.max(all_40_images, dim=1, keepdim=True)  # 结果形状(1, 1, 256, 256)
    
    result = max_pooled_result.squeeze().cpu().numpy()
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    result = Image.fromarray(result)
    result.save(os.path.join(output_dir, 'pred.png'))
    print(f"通道维度Max Pooling结果已保存到：{os.path.join(output_dir, 'pred.png')}")

def split_into_batches(full_list, batch_size=8):
    """将完整列表按指定批次大小拆分成多个子批次（核心分批函数）"""
    batches = []
    for i in range(0, len(full_list), batch_size):
        batch = full_list[i:i+batch_size]
        batches.append(batch)
    return batches

def inference_fm(image_path, checkpoint_path, output_dir, L_min, L_max, cfg_scale_interval=2, batch_size=8, target_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dpm_full_list = list(range(L_min, L_max+1)) + list(range(-L_max, -L_min+1))
    
    dpm_batches = split_into_batches(dpm_full_list, batch_size=batch_size)
    
    original_subdir = os.path.join(output_dir, "pred_z_stacks")  # output_dir下的新建子文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    lq = load_image(image_path, device, target_size=target_size)
    print(f"加载并缩放后的图像形状：{lq.shape}（H={lq.shape[2]}, W={lq.shape[3]}）")
    model = load_model(checkpoint_path, 'fm', device)  # 模型输入尺寸设为256
    
    all_images = [lq]
    
    pbar = tqdm(
        iterable=dpm_batches,
        desc="Processing batches",
        unit="batch",
        total=len(dpm_batches)
    )
    for batch_idx, dpm_batch_values in enumerate(pbar):
        lq_batch, dpm_batch = prepare_batch_input(lq, dpm_batch_values, L_max, device)
        output_batch = run_fm_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval)
        save_original_images(output_batch, dpm_batch_values, original_subdir)
        all_images.append(output_batch)
        
        pbar.set_description(f"Processing batch {batch_idx+1}/{len(dpm_batches)}")
    
    pbar.close()
    
    all_images = torch.cat(all_images, dim=0)
    channel_wise_max_pooling(all_images, output_dir)
    
    print("  共生成40张原始图像和1张Max Pooling汇总结果图")

def inference_jit(image_path, checkpoint_path, output_dir, L_min, L_max, cfg_scale_interval=2, batch_size=8, target_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dpm_full_list = list(range(L_min, L_max+1)) + list(range(-L_max, -L_min+1))
    
    dpm_batches = split_into_batches(dpm_full_list, batch_size=batch_size)
    
    original_subdir = os.path.join(output_dir, "original_images")  # output_dir下的新建子文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    lq = load_image(image_path, device, target_size=target_size)
    print(f"加载并缩放后的图像形状：{lq.shape}（H={lq.shape[2]}, W={lq.shape[3]}）")
    model = load_model(checkpoint_path, 'jit', device)  # 模型输入尺寸设为256
    
    all_images = [lq]
    
    pbar = tqdm(
        iterable=dpm_batches,
        desc="Processing batches",
        unit="batch",
        total=len(dpm_batches)
    )
    for batch_idx, dpm_batch_values in enumerate(pbar):
        lq_batch, dpm_batch = prepare_batch_input(lq, dpm_batch_values, L_max, device)
        output_batch = run_jit_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval)
        save_original_images(output_batch, dpm_batch_values, original_subdir)
        all_images.append(output_batch)
        
        pbar.set_description(f"Processing batch {batch_idx+1}/{len(dpm_batches)}")
    
    pbar.close()
    
    all_images = torch.cat(all_images, dim=0)
    channel_wise_max_pooling(all_images, output_dir)
    
    print("  共生成40张原始图像和1张Max Pooling汇总结果图")

def inference_gan(image_path, checkpoint_path, output_dir, L_min, L_max, cfg_scale_interval=2, batch_size=8, target_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dpm_full_list = list(range(L_min, L_max+1)) + list(range(-L_max, -L_min+1))
    
    dpm_batches = split_into_batches(dpm_full_list, batch_size=batch_size)
    
    original_subdir = os.path.join(output_dir, "original_images")  # output_dir下的新建子文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    lq = load_image(image_path, device, target_size=target_size)
    print(f"加载并缩放后的图像形状：{lq.shape}（H={lq.shape[2]}, W={lq.shape[3]}）")
    model = load_model(checkpoint_path, 'gan', device)  # 模型输入尺寸设为256
    
    all_images = [lq]
    
    pbar = tqdm(
        iterable=dpm_batches,
        desc="Processing batches",
        unit="batch",
        total=len(dpm_batches)
    )
    for batch_idx, dpm_batch_values in enumerate(pbar):
        lq_batch, dpm_batch = prepare_batch_input(lq, dpm_batch_values, L_max, device)
        with torch.no_grad():
            output_batch = model(lq_batch, dpm_batch)
        save_original_images(output_batch, dpm_batch_values, original_subdir)
        all_images.append(output_batch)
        
        pbar.set_description(f"Processing batch {batch_idx+1}/{len(dpm_batches)}")
    
    pbar.close()
    
    all_images = torch.cat(all_images, dim=0)
    channel_wise_max_pooling(all_images, output_dir)

def generate_gt_image(ch, image_idx, z, L_min, L_max, output_dir):
    """
    读取图像并进行max pooling，生成gt.png
    
    Args:
        ch: 通道号
        image_idx: 图像索引
        z: 中心z值
        L_min: 最小偏移量
        L_max: 最大偏移量
        output_dir: 输出目录
    """
    base_path = f'/data1/azt/cv/recoverZ/data_simulation/images/{image_idx}/'

    # 读取中心图像以获取尺寸
    center_image_path = base_path + f'z{z}.png'
    center_img = cv2.imread(center_image_path, cv2.IMREAD_GRAYSCALE)
    if center_img is None:
        print(f"警告：无法读取中心图像: {center_image_path}，使用默认尺寸")
        max_pooled = np.zeros((256, 256), dtype=np.float32)
    else:
        max_pooled = np.zeros_like(center_img, dtype=np.float32)
    
    for k in range(L_min, L_max + 1):
        img_path = base_path + f'z{z-k}.png'
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if max_pooled.size == 0:
                    max_pooled = img.astype(np.float32)
                else:
                    if img.shape == max_pooled.shape:
                        max_pooled = np.maximum(max_pooled, img.astype(np.float32))
                    else:
                        print(f"尺寸不匹配: {img_path} 尺寸 {img.shape} vs {max_pooled.shape}")
    
    for k in range(L_min, L_max + 1):
        img_path = base_path + f'z{z+k}.png'
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape == max_pooled.shape:
                    max_pooled = np.maximum(max_pooled, img.astype(np.float32))
                else:
                    print(f"尺寸不匹配: {img_path} 尺寸 {img.shape} vs {max_pooled.shape}")
    
    max_pooled = np.clip(max_pooled, 0, 255).astype(np.uint8)
    gt_path = os.path.join(output_dir, 'gt.png')
    cv2.imwrite(gt_path, max_pooled)
    print(f"最大池化结果已保存到: {gt_path}")

def stitch_patches(patches, n_rows, n_cols):
    """
    将 n_rows x n_cols 个 patch 按行优先顺序拼回大图。
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


def process_group_inference(data_root, output_root, layer_input, layer_min, layer_max,
                             checkpoint_path, model_type='fm', cfg_scale_interval=2,
                             batch_size=8, target_size=256, K=4, n_rows=2, n_cols=2):
    """
    批量处理 data_root 下的所有数字子文件夹，每K个为一组。
    对于每一组，推理全部子文件夹后按行优先顺序拼接成大图，输出 5 个文件：
      - input.png: 拼接后的输入图像
      - input.tif: 拼接后的 GT z-stack TIF
      - pred.tif: 拼接后的预测 z-stack TIF
      - input_z{z}.png: 随机抽取某一层的拼接 GT
      - pred_z{z}.png: 对应层的拼接预测
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    dpm_raw = [layer_input - t for t in target_layers]
    dpm_max = max(abs(d) for d in dpm_raw)
    num_layers = len(target_layers)

    # 按 K 个一组分组
    groups = [subfolders[i:i + K] for i in range(0, len(subfolders), K)]

    print(f"共 {len(subfolders)} 个子文件夹，每 {K} 个为一组（{n_rows}x{n_cols}），共 {len(groups)} 组")
    print(f"输入层：z{layer_input}，目标层：z{layer_min}~z{layer_max}，DPM范围：{min(dpm_raw)}~{max(dpm_raw)}")

    pbar = tqdm(groups, desc="推理组", unit="group")
    for group_idx, group_folders in enumerate(pbar):
        # 组名：如 "1-4", "5-8"
        start_id = group_folders[0][0]
        end_id = group_folders[-1][0]
        group_name = f"{start_id}-{end_id}"
        group_output = os.path.join(output_root, group_name)
        os.makedirs(group_output, exist_ok=True)

        # 收集该组所有 patch 的推理结果
        # all_group_preds[patch_idx][layer_idx] = (H, W) numpy
        all_group_preds = []       # 每个patch的pred: list of (num_layers, H, W)
        all_group_inputs = []      # 每个patch的input: (H, W) numpy
        all_group_gts = []         # 每个patch的gt z-stack: (num_layers, H, W) numpy

        for folder_id, folder_name in group_folders:
            input_image_path = os.path.join(data_root, folder_name, f"z{layer_input}.png")
            if not os.path.exists(input_image_path):
                print(f"\n  跳过 {folder_name}：z{layer_input}.png 不存在，用零填充")
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

            all_predictions = torch.cat(all_predictions, dim=0)  # (num_layers, 1, H, W)
            pred_np = all_predictions.squeeze(1).cpu().numpy() * 255.0
            pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)  # (num_layers, H, W)
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
            all_group_gts.append(np.stack(gt_stack, axis=0))  # (num_layers, H, W)

        # 不足 K 个的用零填充补齐
        while len(all_group_preds) < K:
            all_group_preds.append(np.zeros((num_layers, target_size, target_size), dtype=np.uint8))
            all_group_inputs.append(np.zeros((target_size, target_size), dtype=np.uint8))
            all_group_gts.append(np.zeros((num_layers, target_size, target_size), dtype=np.uint8))

        # === 拼接并保存 5 个文件 ===

        # 1. input.png（拼接输入层）
        stitched_input = stitch_patches(all_group_inputs, n_rows, n_cols)
        Image.fromarray(stitched_input).save(os.path.join(group_output, "input.png"))

        # 2. pred.tif（拼接预测 z-stack）
        pred_tif_frames = []
        for layer_idx in range(num_layers):
            patches = [all_group_preds[p][layer_idx] for p in range(K)]
            stitched = stitch_patches(patches, n_rows, n_cols)
            pred_tif_frames.append(Image.fromarray(stitched))
        pred_tif_path = os.path.join(group_output, "pred.tif")
        pred_tif_frames[0].save(pred_tif_path, save_all=True,
                                append_images=pred_tif_frames[1:], compression="tiff_adobe_deflate")

        # 3. input.tif（拼接 GT z-stack）
        input_tif_frames = []
        for layer_idx in range(num_layers):
            patches = [all_group_gts[p][layer_idx] for p in range(K)]
            stitched = stitch_patches(patches, n_rows, n_cols)
            input_tif_frames.append(Image.fromarray(stitched))
        input_tif_path = os.path.join(group_output, "input.tif")
        input_tif_frames[0].save(input_tif_path, save_all=True,
                                 append_images=input_tif_frames[1:], compression="tiff_adobe_deflate")

        # 4 & 5. 随机抽取某一层，保存 input_z{z}.png 和 pred_z{z}.png
        random_z = random.choice(target_layers)
        random_idx = target_layers.index(random_z)

        # GT 的该层（拼接）
        gt_patches = [all_group_gts[p][random_idx] for p in range(K)]
        stitched_gt = stitch_patches(gt_patches, n_rows, n_cols)
        Image.fromarray(stitched_gt).save(os.path.join(group_output, f"input_z{random_z}.png"))

        # pred 的该层（拼接）
        pred_patches = [all_group_preds[p][random_idx] for p in range(K)]
        stitched_pred = stitch_patches(pred_patches, n_rows, n_cols)
        Image.fromarray(stitched_pred).save(os.path.join(group_output, f"pred_z{random_z}.png"))

        print(f"  组 {group_name} 完成：input.png, input.tif, pred.tif, input_z{random_z}.png, pred_z{random_z}.png")

    pbar.close()
    print(f"\n处理完成！结果保存在：{output_root}")


if __name__ == "__main__":
    group_id =  ['1', '2', '3']  # 可以是 str（如 '1'）或 list[str]（如 ['1', '2', '3']）
    model_type = "real_plain"
    layer_input = 10
    layer_min, layer_max = 1, 21
    K = 4                # 每 K 个子文件夹为一组
    n_rows = 2           # 拼接行数（K = n_rows * n_cols）
    n_cols = 2           # 拼接列数
    cfg_scale_interval = 2
    batch_size = 8

    network_path = f'/data1/azt/cv/recoverZ/outputs/{model_type}/fm_palette'
    checkpoint_path = os.path.join(network_path, 'checkpoints/best_val_loss_ema.pt')

    # 支持 group_id 为 str 或 list[str]
    group_ids = [group_id] if isinstance(group_id, str) else list(group_id)

    for gid in group_ids:
        print(f"\n{'='*60}")
        print(f"开始处理 group{gid}")
        print(f"{'='*60}")
        data_root = f"/data1/azt/cv/recoverZ/data_test/group{gid}"
        output_root = f"outputs/test/group{gid}"
        os.makedirs(output_root, exist_ok=True)

        process_group_inference(
            data_root=data_root,
            output_root=output_root,
            layer_input=layer_input,
            layer_min=layer_min,
            layer_max=layer_max,
            checkpoint_path=checkpoint_path,
            model_type='fm',
            cfg_scale_interval=cfg_scale_interval,
            batch_size=batch_size,
            target_size=256,
            K=K,
            n_rows=n_rows,
            n_cols=n_cols
        )