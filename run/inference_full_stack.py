import os
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import cv2

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
    cfg = OmegaConf.load("configs/train.yaml")
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

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    image_idx = "5"
    z = 10
    L_min, L_max = 2, 10
    ch = 2
    cfg_scale_interval = 2
    model_name = "fm_palette"
    # input_image_path = f'/data1/azt/cv/recoverZ/data/images_cropped/ch{ch}/{image_idx}/z{z}.png'
    input_image_path = f'/data1/azt/cv/recoverZ/data_simulation/images/{image_idx}/z{z}.png'
    network_path = '/data1/azt/cv/recoverZ/outputs/simulation/fm_palette'
    checkpoint_path = os.path.join(network_path, 'checkpoints/best_val_loss_ema.pt')
    output_dir = os.path.join(network_path, 'inference_z_stacks')
    input_dest_path = os.path.join(output_dir, 'input.png')

    if model_name.startswith("fm"):
        inference_method = inference_fm
    elif model_name.startswith("gan"):
        inference_method = inference_gan
    elif model_name.startswith("jit"):
        inference_method = inference_jit
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
        
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        cv2.imwrite(input_dest_path, img)
    else:
        print(f"Warning: Could not read image from {input_image_path}")
    
    inference_method(
        image_path=input_image_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        L_min=L_min,
        L_max=L_max,
        cfg_scale_interval=cfg_scale_interval,
        batch_size=16,
        target_size=256
    )

    generate_gt_image(ch, image_idx, z, L_min, L_max, output_dir)