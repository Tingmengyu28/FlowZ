import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import trange
import tifffile

random.seed(42)
np.random.seed(42)

num_stacks = 400
num_slices = 21
img_size = 256
save_path = "data_simulation/images"
aif_save_path = "data_simulation/aif_images"
aif_depth_save_path = "data_simulation/aif_depths"
noise_mean = 0.0
noise_sigma = 0.01

psf_path = "data_simulation/PSF_bbbc006.tif"
psf_data = tifffile.imread(psf_path)
print(f"Loaded PSF data with shape: {psf_data.shape}")
psf_mid_layer = psf_data[psf_data.shape[0] // 2, :, :]

os.makedirs(save_path, exist_ok=True)
os.makedirs(aif_save_path, exist_ok=True)
os.makedirs(aif_depth_save_path, exist_ok=True)
txt_save_dir = "data_simulation"
os.makedirs(txt_save_dir, exist_ok=True)
txt_file_path = os.path.join(txt_save_dir, "points_coordinates.txt")

with open(txt_file_path, 'w') as txt_file:
    for stack_idx in trange(1, num_stacks + 1):
        stack_dir = os.path.join(save_path, str(stack_idx))
        os.makedirs(stack_dir, exist_ok=True)
        
        num_points = random.randint(10, 30)
        points = [(random.randint(30, img_size-30), 
                   random.randint(30, img_size-30), 
                   random.randint(0, num_slices-1)) for _ in range(num_points)]
        
        for point_idx, (px, py, pz) in enumerate(points, 1):
            txt_file.write(f"{stack_idx} {px} {py} {pz}\n")
        
        stack_imgs = []
        global_max = 0.0
        for slice_idx in range(num_slices):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            for (px, py, pz) in points:
                distance = slice_idx - pz + psf_data.shape[0] // 2
                distance = min(distance, psf_data.shape[0] - 1)
                psf = psf_data[distance, :, :]
                
                psf_h, psf_w = psf.shape
                start_x = max(0, px - psf_h // 2)
                end_x = min(img_size, px + psf_h // 2 + 1)
                start_y = max(0, py - psf_w // 2)
                end_y = min(img_size, py + psf_w // 2 + 1)
                
                actual_psf_start_x = max(0, psf_h // 2 - px)
                actual_psf_end_x = min(psf.shape[0], actual_psf_start_x + (end_x - start_x))
                actual_psf_start_y = max(0, psf_w // 2 - py)
                actual_psf_end_y = min(psf.shape[1], actual_psf_start_y + (end_y - start_y))
                
                actual_img_start_x = start_x
                actual_img_end_x = start_x + (actual_psf_end_x - actual_psf_start_x)
                actual_img_start_y = start_y
                actual_img_end_y = start_y + (actual_psf_end_y - actual_psf_start_y)
                
                img[actual_img_start_x:actual_img_end_x, actual_img_start_y:actual_img_end_y] += \
                    psf[actual_psf_start_x:actual_psf_end_x, actual_psf_start_y:actual_psf_end_y]
            stack_imgs.append(img)
            current_max = np.max(img)
            if current_max > global_max:
                global_max = current_max
        
        for slice_idx in range(num_slices):
            img = stack_imgs[slice_idx]
            if global_max > 0:
                img = img / global_max
                img *= 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            slice_path = os.path.join(stack_dir, f"z{slice_idx}.png")
            plt.imsave(slice_path, img, cmap='gray', vmin=0, vmax=255)
        
        # 同时生成普通AIF和深度AIF图像
        aif_img = np.zeros((img_size, img_size), dtype=np.float32)
        aif_depth_img = np.zeros((img_size, img_size), dtype=np.float32)
        psf_h, psf_w = psf_mid_layer.shape
        for (px, py, pz) in points:
            # 使用pz值作为深度信息，将其映射到像素强度
            depth_intensity = (pz / (num_slices - 1)) * 255.0  # 将深度值标准化到0-255
            
            start_x = max(0, px - psf_h // 2)
            end_x = min(img_size, px + psf_h // 2 + 1)
            start_y = max(0, py - psf_w // 2)
            end_y = min(img_size, py + psf_w // 2 + 1)
            
            actual_psf_start_x = max(0, psf_h // 2 - px)
            actual_psf_end_x = min(psf_mid_layer.shape[0], actual_psf_start_x + (end_x - start_x))
            actual_psf_start_y = max(0, psf_w // 2 - py)
            actual_psf_end_y = min(psf_mid_layer.shape[1], actual_psf_start_y + (end_y - start_y))
            
            actual_img_start_x = start_x
            actual_img_end_x = start_x + (actual_psf_end_x - actual_psf_start_x)
            actual_img_start_y = start_y
            actual_img_end_y = start_y + (actual_psf_end_y - actual_psf_start_y)
            
            # 构建普通AIF图像
            psf_chunk = psf_mid_layer[actual_psf_start_x:actual_psf_end_x, actual_psf_start_y:actual_psf_end_y]
            aif_img[actual_img_start_x:actual_img_end_x, actual_img_start_y:actual_img_end_y] += psf_chunk
            
            # 构建深度AIF图像
            aif_depth_img[actual_img_start_x:actual_img_end_x, actual_img_start_y:actual_img_end_y] += \
                psf_chunk * (depth_intensity / 255.0)
        
        # 保存普通AIF图像
        if global_max > 0:
            aif_img = aif_img / global_max
            aif_img *= 255.0
        aif_img = np.clip(aif_img, 0, 255).astype(np.uint8)
        aif_img_path = os.path.join(aif_save_path, f"{stack_idx}.png")
        plt.imsave(aif_img_path, aif_img, cmap='gray', vmin=0, vmax=255)
        
        # 保存深度AIF图像
        if global_max > 0:
            aif_depth_img = aif_depth_img / global_max
            aif_depth_img *= 255.0
        aif_depth_img = np.clip(aif_depth_img, 0, 255).astype(np.uint8)
        aif_depth_img_path = os.path.join(aif_depth_save_path, f"{stack_idx}.png")
        plt.imsave(aif_depth_img_path, aif_depth_img, cmap='gray', vmin=0, vmax=255)
        
        # 归一化深度图像
        if global_max > 0:
            aif_depth_img = aif_depth_img / global_max
            aif_depth_img *= 255.0
        aif_depth_img = np.clip(aif_depth_img, 0, 255).astype(np.uint8)
        aif_depth_img_path = os.path.join(aif_depth_save_path, f"{stack_idx}.png")
        plt.imsave(aif_depth_img_path, aif_depth_img, cmap='gray', vmin=0, vmax=255)

print("All stacks generated")
print(f"Points coordinates saved to: {txt_file_path}")
print(f"AiF images saved to: {aif_save_path}")
print(f"Depth AiF images saved to: {aif_depth_save_path}")