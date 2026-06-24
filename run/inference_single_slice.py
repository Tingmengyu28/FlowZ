import os
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import math
import cv2
import shutil
from PIL import ImageDraw, ImageFont, Image
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
    
    image = F.interpolate(
        image,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )
    
    return image

def load_model(checkpoint_path, model_type, device):
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
    model.load_state_dict(checkpoint["model"])
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
        dpm_single = torch.full((1, H, W), fill_value=value, dtype=torch.float32, device=device)
        dpm_batch.append(dpm_single)
    dpm_batch = torch.stack(dpm_batch, dim=0)
    
    return lq_batch, dpm_batch

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
            v_uncond = model(gen_sample, t_current, torch.zeros_like(cond, device=device, requires_grad=False))
            v_final = cfg_scale_interval * v_cond + (1 - cfg_scale_interval) * v_uncond

            gen_sample = gen_sample + v_final * dt
    
    return gen_sample

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

def calculate_vol(image_array):
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float32)
    
    laplacian = cv2.filter2D(image_array, -1, laplacian_kernel)
    
    vol_value = np.var(laplacian)
    
    return vol_value

def save_original_images(output=None, dpm_batch_values=None, save_dir=None, is_batch_save=True, pil_images=None):
    if is_batch_save:
        if output is None or dpm_batch_values is None:
            print("警告：批次模式需传入output和dpm_batch_values，返回空列表")
            return []
        batch_pil = []
        for i, dpm_value in enumerate(dpm_batch_values):
            result = output[i].squeeze().cpu().numpy()
            result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            
            vol_value = calculate_vol(result).round(2)
            
            img = Image.fromarray(result)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default(size=20)
            
            text = f"z: {dpm_value}, VoL: {vol_value}"
            text_xy = (5, 5)
            text_color = 255 if img.mode == 'L' else (255, 255, 255)
            draw.text(text_xy, text, fill=text_color, font=font)
            single_img_path = os.path.join(save_dir, f"z50_dpm_{dpm_value}.png")
            img.save(single_img_path)
            batch_pil.append(img)
        return batch_pil
    else:
        if pil_images is None or dpm_batch_values is None or len(pil_images) != len(dpm_batch_values):
            print("警告：拼接模式需传入有效pil_images和等长的dpm_batch_values，跳过拼接")
            return
        if not pil_images:
            print("警告：无图像可拼接，跳过保存")
            return
        cols = 5
        total_imgs = len(pil_images)
        rows = math.ceil(total_imgs / cols)
        w, h = pil_images[0].width, pil_images[0].height
        merged_w = w * cols
        merged_h = h * rows
        merged_img = Image.new(pil_images[0].mode, (merged_w, merged_h))
        for i, img in enumerate(pil_images):
            row = i // cols
            col = i % cols
            x = col * w
            y = row * h
            merged_img.paste(img, (x, y))
        merged_path = os.path.join(save_dir, "merged_all_dpm_images")
        file_name_noext, ext = os.path.splitext(merged_path)
        valid_exts = ['.png', '.jpg', '.jpeg']
        if ext.lower() not in valid_exts:
            merged_path = f"{file_name_noext}.png"
        merged_img.save(merged_path)
        print(f"✅ 所有dpm图像拼接完成，完整大图已保存至：{merged_path}")
        print(f"✅ 所有单张dpm图像已保存至：{os.path.join(save_dir, 'original_images')}")

def split_into_batches(full_list, batch_size=8):
    batches = []
    for i in range(0, len(full_list), batch_size):
        batch = full_list[i:i+batch_size]
        batches.append(batch)
    return batches

def inference_fm(image_path, checkpoint_path, output_dir, z, L_min, L_max, cfg_scale_interval=2, batch_size=8, target_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dpm_full_list = list(range(L_min, L_max+1))
    dpm_full_list = [dpm + z for dpm in dpm_full_list]
    dpm_batches = split_into_batches(dpm_full_list, batch_size=batch_size)
    
    original_subdir = os.path.join(output_dir, "z-stack")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(original_subdir, exist_ok=True)
    
    all_pil_images = []
    all_dpm_values = []
    
    lq = load_image(image_path, device, target_size=target_size)
    print(f"加载并缩放后的图像形状：{lq.shape}（H={lq.shape[2]}, W={lq.shape[3]}）")
    model = load_model(checkpoint_path, 'fm', device)
    
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
        
        batch_pil_images = save_original_images(
            output=output_batch,
            dpm_batch_values=dpm_batch_values,
            save_dir=original_subdir,
            is_batch_save=True
        )
        all_pil_images.extend(batch_pil_images)
        all_dpm_values.extend(dpm_batch_values)
        
        all_images.append(output_batch)
        pbar.set_description(f"Processing batch {batch_idx+1}/{len(dpm_batches)}")
    
    pbar.close()
    
    if all_pil_images and all_dpm_values:
        save_original_images(
            output=None,
            dpm_batch_values=all_dpm_values,
            save_dir=output_dir,
            is_batch_save=False,
            pil_images=all_pil_images
        )
    
    all_images = torch.cat(all_images, dim=0)
    
    total_original = len(dpm_full_list)
    print(f"  共生成{total_original}张原始图像（保存至{original_subdir}）和1张完整拼接大结果图（保存至{output_dir}）")

def inference_jit(image_path, checkpoint_path, output_dir, z, L_min, L_max, cfg_scale_interval=2, batch_size=8, target_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dpm_full_list = list(range(L_min, L_max+1))
    
    dpm_batches = split_into_batches(dpm_full_list, batch_size=batch_size)
    
    original_subdir = os.path.join(output_dir, "original_images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(original_subdir, exist_ok=True)
    
    all_pil_images = []
    all_dpm_values = []
    
    lq = load_image(image_path, device, target_size=target_size)
    print(f"加载并缩放后的图像形状：{lq.shape}（H={lq.shape[2]}, W={lq.shape[3]}）")
    model = load_model(checkpoint_path, 'jit', device)
    
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
        
        batch_pil_images = save_original_images(
            output=output_batch,
            dpm_batch_values=dpm_batch_values,
            save_dir=original_subdir,
            is_batch_save=True
        )
        all_pil_images.extend(batch_pil_images)
        all_dpm_values.extend(dpm_batch_values)
        
        all_images.append(output_batch)
        pbar.set_description(f"Processing batch {batch_idx+1}/{len(dpm_batches)}")
    
    pbar.close()
    
    if all_pil_images and all_dpm_values:
        save_original_images(
            output=None,
            dpm_batch_values=all_dpm_values,
            save_dir=output_dir,
            is_batch_save=False,
            pil_images=all_pil_images
        )
    
    all_images = torch.cat(all_images, dim=0)
    
    total_original = len(dpm_full_list)
    print(f"  共生成{total_original}张原始图像（保存至{original_subdir}）和1张完整拼接大结果图（保存至{output_dir}）")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    image_name = "a10"
    z = 3
    L_min, L_max = 2, 21
    s = 2
    cfg_scale_interval = 2
    model_name = "fm_palette"
    input_image_path = f'/data1/azt/cv/recoverZ/data_BBBC006/images/s{s}/{image_name}/z{z}.png'
    network_path = '/data1/azt/cv/recoverZ/outputs/train/BBBC006/fm_palette'
    checkpoint_path = os.path.join(network_path, 'checkpoints/best_val_loss_ema.pt')
    output_dir = os.path.join(network_path, 'inference_details')
    input_dest_path = os.path.join(output_dir, 'input.png')

    if model_name.startswith("fm"):
        inference_method = inference_fm
    elif model_name.startswith("jit"):
        inference_method = inference_jit
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(input_dest_path, img)

    inference_method(
        image_path=input_image_path,
        checkpoint_path=checkpoint_path ,
        output_dir=output_dir,
        z=z,
        L_min=L_min,
        L_max=L_max,
        cfg_scale_interval=cfg_scale_interval,
        batch_size=8,
        target_size=256
    )

    shutil.copy2(f'/data1/azt/cv/recoverZ/data_BBBC006/images/s{s}/{image_name}/z16.png',
                 f'/data1/azt/cv/recoverZ/outputs/inference/{model_name}/s{s}/{image_name}/')