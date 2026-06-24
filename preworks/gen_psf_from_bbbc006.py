import os
import numpy as np
import cv2
from scipy import interpolate
from tifffile import imwrite

def read_z_stack_images(folder_path, crop_x, crop_y, crop_size):
    z_files = []
    for i in range(1, 35):
        file_path = os.path.join(folder_path, f"z{i}.png")
        if os.path.exists(file_path):
            z_files.append(file_path)
    
    if len(z_files) != 34:
        raise ValueError(f"未找到全部34张图像，仅找到{len(z_files)}张")
    
    z_stack = []
    for file_path in z_files:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {file_path}")
        
        cropped_img = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
        z_stack.append(cropped_img)
    
    return np.array(z_stack, dtype=np.uint16)

def interpolate_z_stack(z_stack, target_layers=45):
    original_z = z_stack.shape[0]
    height, width = z_stack.shape[1], z_stack.shape[2]
    
    original_coords = np.linspace(0, 1, original_z)
    target_coords = np.linspace(0, 1, target_layers)
    
    interpolated_stack = np.zeros((target_layers, height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            pixel_vals = z_stack[:, y, x]
            f = interpolate.interp1d(original_coords, pixel_vals, kind='linear', fill_value="extrapolate")
            interpolated_stack[:, y, x] = f(target_coords)
    
    interpolated_stack = np.clip(interpolated_stack, 0, 65535).astype(np.uint16)
    
    return interpolated_stack

def apply_threshold_filter(stack_data, threshold=200):
    mask = stack_data < threshold
    filtered_stack = stack_data.copy()
    filtered_stack[mask] = 0
    
    total_pixels = filtered_stack.size
    zero_pixels = np.sum(mask)
    print(f"阈值滤波完成 - 阈值: {threshold}, 置0像素数: {zero_pixels}/{total_pixels} ({zero_pixels/total_pixels*100:.2f}%)")
    
    return filtered_stack

def apply_radial_mask(stack_data, R_MIN=5, R_MAX=20):
    total_layers = stack_data.shape[0]
    height, width = stack_data.shape[1], stack_data.shape[2]
    center_y, center_x = height // 2, width // 2
    center_layer = total_layers // 2
    
    masked_stack = stack_data.copy()
    
    y_coords, x_coords = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    for layer_idx in range(total_layers):
        distance_from_center_layer = abs(layer_idx - center_layer)
        max_distance = total_layers // 2
        
        if max_distance == 0:
            r = R_MIN
        else:
            r = R_MIN + (distance_from_center_layer / max_distance) * (R_MAX - R_MIN)
            r = np.clip(r, R_MIN, R_MAX)
        
        mask = dist_from_center > r
        masked_stack[layer_idx, mask] = 0
        
        if layer_idx % 10 == 0:
            print(f"第{layer_idx+1}层 - 半径: {r:.2f}")
    
    print(f"径向掩码完成 - 最小半径: {R_MIN}, 最大半径: {R_MAX}")
    return masked_stack

def normalize_stack_global(stack_data):
    global_min = np.min(stack_data)
    global_max = np.max(stack_data)
    
    if global_max == global_min:
        normalized_stack = np.zeros_like(stack_data, dtype=np.uint8)
    else:
        normalized_stack = ((stack_data - global_min) / (global_max - global_min) * 255).astype(np.uint8)
    
    print(f"全局归一化参数 - 最小值: {global_min}, 最大值: {global_max}")
    return normalized_stack

def apply_gamma_correction(normalized_stack, gamma=1.0):
    inv_gamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 
                           for i in np.arange(0, 256)]).astype(np.uint8)
    
    gamma_corrected_stack = np.zeros_like(normalized_stack)
    for i in range(normalized_stack.shape[0]):
        gamma_corrected_stack[i] = cv2.LUT(normalized_stack[i], gamma_table)
    
    print(f"已应用Gamma变换，Gamma值: {gamma}")
    return gamma_corrected_stack

def main():
    img_folder = "data_BBBC006/images/s2/a10"
    crop_x = 623
    crop_y = 190
    crop_size = 45
    output_tif = "data_simulation/PSF_bbbc006.tif"
    layers_output_dir = "data_simulation/psf_layers"
    gamma_value = 1.0
    
    filter_mode = "radius"
    threshold_value = 0
    R_MIN = 3
    R_MAX = 15
    
    os.makedirs(layers_output_dir, exist_ok=True)
    
    try:
        print("读取并裁剪图像...")
        z_stack = read_z_stack_images(img_folder, crop_x, crop_y, crop_size)
        print(f"原始数据形状: {z_stack.shape}")
        
        print("进行z轴插值...")
        interpolated_stack = interpolate_z_stack(z_stack, target_layers=45)
        print(f"插值后数据形状: {interpolated_stack.shape}")
        
        if filter_mode == "threshold":
            print("应用阈值滤波（小于阈值的像素置0）...")
            filtered_stack = apply_threshold_filter(interpolated_stack, threshold=threshold_value)
        elif filter_mode == "radius":
            print("应用径向圆形掩码...")
            filtered_stack = apply_radial_mask(interpolated_stack, R_MIN=R_MIN, R_MAX=R_MAX)
        else:
            raise ValueError(f"不支持的筛选模式: {filter_mode}，可选值: threshold/radius")
        
        print("保存筛选后的TIFF文件...")
        imwrite(output_tif, filtered_stack, photometric='minisblack')
        
        print("对筛选后的TIFF数据进行全局归一化...")
        normalized_stack = normalize_stack_global(filtered_stack)
        
        print("应用Gamma变换增强对比度...")
        gamma_corrected_stack = apply_gamma_correction(normalized_stack, gamma=gamma_value)
        
        print("保存处理后的PNG文件...")
        total_layers = interpolated_stack.shape[0]
        for i in range(total_layers):
            layer_idx = i
            layer_filename = f"z{i+1}.png"
            layer_path = os.path.join(layers_output_dir, layer_filename)
            
            layer_data = gamma_corrected_stack[layer_idx, :, :]
            cv2.imwrite(layer_path, layer_data)
            
            if (i+1) % 10 == 0:
                print(f"已保存 {i+1}/{total_layers} 层")
        
        print("处理完成！")
        print(f"- 筛选后的TIFF文件已保存至: {output_tif}")
        print(f"- 处理后的{total_layers}层PNG已保存至: {layers_output_dir}")
        print(f"- 使用的筛选模式: {filter_mode}, Gamma值: {gamma_value}")
        if filter_mode == "threshold":
            print(f"- 使用的阈值: {threshold_value}")
        else:
            print(f"- 使用的最小半径: {R_MIN}, 最大半径: {R_MAX}")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()