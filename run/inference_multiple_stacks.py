import os
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import cv2
import csv
import matplotlib.pyplot as plt
import matplotlib

import sys
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils.common import instantiate_from_config  # noqa: E402


def get_device():
    """获取设备，优先使用GPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

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

def save_original_images(output, dpm_batch_values, original_subdir):
    os.makedirs(original_subdir, exist_ok=True)
    for i, value in enumerate(dpm_batch_values):
        result = output[i].squeeze().cpu().numpy()
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        result = Image.fromarray(result)
        result.save(os.path.join(original_subdir, f'z50_dpm_{value}.png'))

def channel_wise_max_pooling(image_idx, all_images, output_dir):
    all_images = all_images.squeeze(dim=1)
    if image_idx == '5':
        for i, image in enumerate(all_images):
            os.makedirs(os.path.join(output_dir, 'z-stack'), exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, 'z-stack', f'z{i}.png'), (image * 255).cpu().numpy())

    all_images = all_images.unsqueeze(dim=0)
    
    max_pooled_result, max_z_indices = torch.max(all_images, dim=1, keepdim=True)
    
    result = max_pooled_result.squeeze().cpu().numpy()
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    result = Image.fromarray(result)
    os.makedirs(os.path.join(output_dir, 'pred'), exist_ok=True)
    result.save(os.path.join(output_dir, 'pred',f'{image_idx}.png'))
    
    all_images_np = all_images.squeeze(0).cpu().numpy()
    max_intensity = np.max(all_images_np)
    threshold = THRESHOLD * max_intensity
    
    max_z_indices_np = max_z_indices.squeeze().cpu().numpy().astype(np.uint8)
    
    max_pooled_result_np = max_pooled_result.squeeze().cpu().numpy()
    mask = max_pooled_result_np > threshold
    depth_pred = np.zeros_like(max_z_indices_np, dtype=np.uint8)
    depth_pred[mask] = max_z_indices_np[mask]
    
    os.makedirs(os.path.join(output_dir, 'depth_pred'), exist_ok=True)
    depth_pred_path = os.path.join(output_dir, 'depth_pred', f'{image_idx}.png')
    depth_pred *= 10
    cv2.imwrite(depth_pred_path, depth_pred)
    
    loc_csv_path = os.path.join(output_dir, 'loc.csv')
    file_exists = os.path.exists(loc_csv_path)
    
    with open(loc_csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(['frame', 'x', 'y', 'z', 'intensity'])
        
        H, W = all_images_np.shape[-2], all_images_np.shape[-1]
        max_z_indices_np = max_z_indices.squeeze().cpu().numpy()
        
        for y in range(H):
            for x in range(W):
                z = max_z_indices_np[y, x]
                intensity = all_images_np[z, y, x]
                
                if intensity > threshold:
                    csv_writer.writerow([image_idx, x, y, z, round(float(intensity), 6)])

def split_into_batches(full_list, batch_size=8):
    batches = []
    for i in range(0, len(full_list), batch_size):
        batch = full_list[i:i+batch_size]
        batches.append(batch)
    return batches

def inference_fm(image_idx, image_path, checkpoint_path, output_dir, L_min, L_max, z, S_min, S_max, cfg_scale_interval=2, batch_size=8, target_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dpm_start = max(-L_max, S_min - z)
    dpm_end = min(L_max, S_max - z)
    
    negative_part = list(range(-L_max, -L_min+1))
    positive_part = list(range(L_min, L_max+1))
    
    dpm_full_list = [d for d in negative_part + positive_part if dpm_start <= d <= dpm_end]
    
    dpm_batches = split_into_batches(dpm_full_list, batch_size=batch_size)
    
    os.makedirs(output_dir, exist_ok=True)
    
    lq = load_image(image_path, device, target_size=target_size)
    model = load_model(checkpoint_path, 'fm', device)
    
    all_images = []
    
    pbar = tqdm(iterable=dpm_batches, desc="Processing batches", unit="batch", total=len(dpm_batches))
    for batch_idx, dpm_batch_values in enumerate(pbar):
        lq_batch, dpm_batch = prepare_batch_input(lq, dpm_batch_values, L_max, device)
        output_batch = run_fm_inference_batch(model, lq_batch, dpm_batch, cfg_scale_interval)
        
        batch_max = output_batch.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        threshold_values = torch.maximum(torch.tensor(MIN_VALID_VALUE / 255.0).to(output_batch.device), THRESHOLD * batch_max)
        output_batch = torch.where(output_batch < threshold_values, torch.tensor(0.0).to(output_batch.device), output_batch)
        
        all_images.append(output_batch)
        
        pbar.set_description(f"Processing batch {batch_idx+1}/{len(dpm_batches)}")
    
    pbar.close()
    lq_repeated = lq.repeat(3, 1, 1, 1)
    all_images = torch.cat(all_images, dim=0)
    all_images = torch.cat([all_images[:all_images.shape[0]//2], lq_repeated, all_images[all_images.shape[0]//2:]], dim=0)
    channel_wise_max_pooling(image_idx, all_images, output_dir)
    generate_gt_depth_map(image_idx, output_dir, data_name)

def generate_gt_image(image_idx, output_dir, data_name):
    if data_name == "simulation":
        aif_base_path = f'/data1/azt/cv/recoverZ/data_{data_name}/aif_images/'
        aif_img_path = aif_base_path + f'{image_idx}.png'
        
        aif_img = cv2.imread(aif_img_path, cv2.IMREAD_GRAYSCALE)
        if aif_img is None:
            print(f"警告：无法读取AiF图像: {aif_img_path}，生成空图像")
            aif_img = np.zeros((256, 256), dtype=np.uint8)
        
        os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
        gt_path = os.path.join(output_dir, 'gt', f'{image_idx}.png')
        cv2.imwrite(gt_path, aif_img)
    else:
        image_dir = f'/data1/azt/cv/recoverZ/data_{data_name}/images/{image_idx}'
        
        z_images = []
        z_idx = 1
        while True:
            img_path = os.path.join(image_dir, f'z{z_idx}.png')
            if not os.path.exists(img_path):
                break
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            threshold_value = max(MIN_VALID_VALUE, THRESHOLD * img.max())
            img = np.where(img < threshold_value, 0, img)

            if img is not None:
                z_images.append(img)
            else:
                print(f"警告：无法读取图像: {img_path}")
            z_idx += 1
        
        z_tensors = [torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() for img in z_images]
        z_stack = torch.cat(z_tensors, dim=0)
        aif_img = torch.max(z_stack, dim=0)[0].squeeze().numpy()
        
        # threshold_value = max(100, THRESHOLD * aif_img.max())
        # aif_img = np.where(aif_img < threshold_value, 0, aif_img)
        
        aif_img = aif_img.astype(np.uint8)
        
        os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
        gt_path = os.path.join(output_dir, 'gt', f'{image_idx}.png')
        cv2.imwrite(gt_path, aif_img)

def generate_gt_depth_map(image_idx, output_dir, data_name):
    image_dir = f'/data1/azt/cv/recoverZ/data_{data_name}/images/{image_idx}'
    
    z_images = []
    z_idx = 0
    while True:
        img_path = os.path.join(image_dir, f'z{z_idx}.png')
        if not os.path.exists(img_path):
            break
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            z_images.append(img)
        else:
            print(f"警告：无法读取图像: {img_path}")
        z_idx += 1
    
    if len(z_images) == 0:
        print(f"警告：未找到任何图像: {image_dir}")
        return
    
    all_images = np.stack(z_images, axis=0)  # (num_z_layers, H, W)
    all_images = all_images.astype(np.float32) / 255.0
    
    max_pooled_result = np.max(all_images, axis=0)  # (H, W)
    max_z_indices = np.argmax(all_images, axis=0)   # (H, W)
    
    max_intensity = np.max(all_images)
    threshold = THRESHOLD * max_intensity
    
    depth_map = max_z_indices.astype(np.uint8)
    
    mask = max_pooled_result > threshold
    depth_filtered = np.zeros_like(depth_map, dtype=np.uint8)
    depth_filtered[mask] = depth_map[mask]
    
    os.makedirs(os.path.join(output_dir, 'depth_gt'), exist_ok=True)
    depth_path = os.path.join(output_dir, 'depth_gt', f'{image_idx}.png')
    depth_filtered *= 10
    cv2.imwrite(depth_path, depth_filtered)

def calculate_metrics_for_pairs(output_dir, image_indices):
    print("\n" + "="*80)
    print("开始计算PSNR和SSIM指标")
    print("="*80)
    
    print(f"{'Image Index':<15} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-"*80)
    
    total_psnr = 0.0
    total_ssim = 0.0
    valid_count = 0
    
    for idx in image_indices:
        pred_path = os.path.join(output_dir, 'pred', f'{idx}.png')
        gt_path = os.path.join(output_dir, 'gt', f'{idx}.png')
        
        if not os.path.exists(pred_path):
            print(f"{idx:<15} {'File missing':<15} {'File missing':<15}")
            print(f"  错误：pred文件不存在: {pred_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"{idx:<15} {'File missing':<15} {'File missing':<15}")
            print(f"  错误：gt文件不存在: {gt_path}")
            continue
        
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if pred_img is None:
            print(f"{idx:<15} {'Read error':<15} {'Read error':<15}")
            print(f"  错误：无法读取pred图像: {pred_path}")
            continue
        if gt_img is None:
            print(f"{idx:<15} {'Read error':<15} {'Read error':<15}")
            print(f"  错误：无法读取gt图像: {gt_path}")
            continue
        
        pred_img_float = pred_img.astype(np.float32)

        if pred_img.shape != gt_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            pred_img_float = cv2.resize(pred_img_float, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        psnr_val = psnr(gt_img, pred_img_float, data_range=255)
        ssim_val = ssim(gt_img, pred_img_float, data_range=255)
        
        print(f"{idx:<15} {psnr_val:<15.4f} {ssim_val:<15.6f}")
        
        total_psnr += psnr_val
        total_ssim += ssim_val
        valid_count += 1
    
    print("-"*80)
    if valid_count > 0:
        avg_psnr = total_psnr / valid_count
        avg_ssim = total_ssim / valid_count
        print(f"{'Average':<15} {avg_psnr:<15.4f} {avg_ssim:<15.6f}")
    else:
        print(f"{'Average':<15} {'N/A':<15} {'N/A':<15}")
    print("="*80)
    
    print("\n=== 指标计算汇总 ===")
    print(f"总处理图像数量: {len(image_indices)}")
    print(f"有效计算数量: {valid_count}")
    if valid_count > 0:
        print(f"平均PSNR: {avg_psnr:.4f} dB")
        print(f"平均SSIM: {avg_ssim:.6f}")
    print("="*80)

def create_comparison_grid(output_dir, image_indices, img_size=256):
    print("\n开始生成对比网格图...")
    
    input_images = []
    gt_images = []
    pred_images = []
    
    for idx in image_indices:
        input_path = os.path.join(output_dir, 'input', f'{idx}.png')
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if input_img is None:
            print(f"警告：无法读取input图像 {idx}，使用空图像替代")
            input_img = np.zeros((img_size, img_size), dtype=np.uint8)
        else:
            input_img = cv2.resize(input_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        gt_path = os.path.join(output_dir, 'gt', f'{idx}.png')
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            print(f"警告：无法读取gt图像 {idx}，使用空图像替代")
            gt_img = np.zeros((img_size, img_size), dtype=np.uint8)
        else:
            gt_img = cv2.resize(gt_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        pred_path = os.path.join(output_dir, 'pred', f'{idx}.png')
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred_img is None:
            print(f"警告：无法读取pred图像 {idx}，使用空图像替代")
            pred_img = np.zeros((img_size, img_size), dtype=np.uint8)
        else:
            pred_img = cv2.resize(pred_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        input_images.append(input_img)
        gt_images.append(gt_img)
        pred_images.append(pred_img)
    
    num_cols = len(image_indices)
    grid_width = num_cols * img_size
    grid_height = 3 * img_size
    
    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    for col, idx in enumerate(range(num_cols)):
        x_start = col * img_size
        x_end = x_start + img_size
        y_start = 0 * img_size
        y_end = y_start + img_size
        grid[y_start:y_end, x_start:x_end] = input_images[idx]
        y_start = 1 * img_size
        y_end = y_start + img_size
        grid[y_start:y_end, x_start:x_end] = gt_images[idx]
        y_start = 2 * img_size
        y_end = y_start + img_size
        grid[y_start:y_end, x_start:x_end] = pred_images[idx]
    
    grid = 255 - grid
    
    for col in range(1, num_cols):
        x_start = col * img_size
        grid[:, x_start:x_start+2] = 0
    
    for row in range(1, 3):
        y_start = row * img_size
        grid[y_start:y_start+2, :] = 0
        
    grid[0:2, :] = 0
    grid[-2:, :] = 0
    grid[:, 0:2] = 0
    grid[:, -2:] = 0
    
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(max(12, num_cols * 2), 9))
    ax.imshow(grid, cmap='gray')
    ax.axis('off')
    
    row_titles = ['Input', 'GT AiF', 'Pred AiF']
    for i, title in enumerate(row_titles):
        ax.text(-20, i * img_size + img_size // 2, title, 
                verticalalignment='center', horizontalalignment='right',
                fontsize=12, fontweight='normal',
                transform=ax.transData)
    
    grid_path = os.path.join(output_dir, 'aif_grid.png')
    plt.savefig(grid_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"对比网格图已保存至: {grid_path}")

def create_3d_scatter_plot(output_dir, data_name, image_indices):
    print("\n开始生成3D散点图及俯视图...")
    
    loc_csv_path = os.path.join(output_dir, 'loc.csv')
    gt_points_path = f'/data1/azt/cv/recoverZ/data_{data_name}/points_coordinates.txt'
    
    pred_data = {}
    if os.path.exists(loc_csv_path):
        with open(loc_csv_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) >= 4:
                    img_idx = row[0]
                    if img_idx not in pred_data:
                        pred_data[img_idx] = {'x': [], 'y': [], 'z': []}
                        pred_data[img_idx]['x'].append(float(row[2]))
                        pred_data[img_idx]['y'].append(float(row[1]))
                        pred_data[img_idx]['z'].append(float(row[3]))
    
    gt_data = {}
    if os.path.exists(gt_points_path):
        with open(gt_points_path, 'r') as txtfile:
            for line in txtfile:
                parts = line.strip().split()
                if len(parts) >= 4:
                    img_idx = parts[0]
                    if img_idx not in gt_data:
                        gt_data[img_idx] = {'x': [], 'y': [], 'z': []}
                    gt_data[img_idx]['x'].append(float(parts[1]))
                    gt_data[img_idx]['y'].append(float(parts[2]))
                    gt_data[img_idx]['z'].append(float(parts[3]))
    
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    num_images = len(image_indices)
    cols = 6
    rows = num_images
    
    fig = plt.figure(figsize=(cols * 3, rows * 4))
    
    for idx, target_image_idx in enumerate(image_indices):
        ax_3d = fig.add_subplot(rows, cols, idx * cols + 1, projection='3d')
        
        pred_x = pred_data.get(target_image_idx, {}).get('x', [])
        pred_y = pred_data.get(target_image_idx, {}).get('y', [])
        pred_z = pred_data.get(target_image_idx, {}).get('z', [])
        
        gt_x = gt_data.get(target_image_idx, {}).get('x', [])
        gt_y = gt_data.get(target_image_idx, {}).get('y', [])
        gt_z = gt_data.get(target_image_idx, {}).get('z', [])
        
        if pred_x:
            ax_3d.scatter(pred_x, pred_y, pred_z, c='blue', s=8, alpha=0.6, label='Pred')
        if gt_x:
            ax_3d.scatter(gt_x, gt_y, gt_z, c='red', s=15, alpha=0.8, label='GT')
        
        ax_3d.set_xlabel('X', fontsize=8)
        ax_3d.set_ylabel('Y', fontsize=8)
        ax_3d.set_zlabel('Z', fontsize=8)
        ax_3d.set_title(f'Image {target_image_idx} - 3D View', fontsize=10, fontweight='bold')
        ax_3d.legend(fontsize=8, loc='upper right')
        ax_3d.tick_params(axis='both', labelsize=6)
        
        ax_top = fig.add_subplot(rows, cols, idx * cols + 2)
        
        if pred_x:
            ax_top.scatter(pred_x, pred_y, c='blue', s=8, alpha=0.6, label='Pred')
        if gt_x:
            ax_top.scatter(gt_x, gt_y, c='red', s=15, alpha=0.8, label='GT')
        
        ax_top.set_xlabel('X', fontsize=8)
        ax_top.set_ylabel('Y', fontsize=8)
        ax_top.set_title(f'Image {target_image_idx} - XY Top View', fontsize=10, fontweight='bold')
        ax_top.legend(fontsize=8, loc='upper right')
        ax_top.tick_params(axis='both', labelsize=6)
        ax_top.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, '3d_scatter_with_topview_all.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"包含俯视图的3D散点图已保存至: {scatter_path}")

if __name__ == "__main__":

    image_idices = [str(idx) for idx in range(1, 5)]
    z = 2
    L_min, L_max = 5, 25
    S_min, S_max = 1, 51
    ch = 2
    cfg_scale_interval = 2
    THRESHOLD = 0.0
    MIN_VALID_VALUE = 0
    model_name = "fm_palette"
    data_name = "real_plain"

    network_path = f'/data1/azt/cv/recoverZ/outputs/{data_name}/fm_palette'
    checkpoint_path = os.path.join(network_path, 'checkpoints/best_val_loss_ema.pt')
    output_dir = os.path.join(network_path, 'inference_details')
    os.makedirs(os.path.join(output_dir, 'input'), exist_ok=True)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    device = get_device()  # 获取设备
    print(f"使用设备: {device}")

    for image_idx in image_idices:
        input_image_path = f'/data1/azt/cv/recoverZ/data_{data_name}/images/{image_idx}/z{z}.png'
        input_dest_path = os.path.join(output_dir, 'input', f'{image_idx}.png')

        if model_name.startswith("fm"):
            inference_method = inference_fm
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
            
        img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            cv2.imwrite(input_dest_path, img)
        else:
            print(f"Warning: Could not read image from {input_image_path}")
        
        inference_method(
            image_idx=image_idx,
            image_path=input_image_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            L_min=L_min,
            L_max=L_max,
            z=z,
            S_min=S_min,
            S_max=S_max,
            cfg_scale_interval=cfg_scale_interval,
            batch_size=16,
            target_size=256
        )

        generate_gt_image(image_idx, output_dir, data_name)
    
    calculate_metrics_for_pairs(output_dir, image_idices)

    create_comparison_grid(output_dir, image_idices, img_size=256)

    if data_name == "simulation":
        create_3d_scatter_plot(output_dir, data_name, image_idices)
