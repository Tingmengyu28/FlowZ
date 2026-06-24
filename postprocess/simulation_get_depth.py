import os
import numpy as np
import cv2

def read_z_stack_images(folder_path):
    z_files = []
    file_list = sorted(os.listdir(folder_path))
    for file_name in file_list:
        if file_name.startswith("z") and file_name.endswith(".png"):
            z_files.append(os.path.join(folder_path, file_name))
    
    if not z_files:
        raise ValueError(f"未找到任何z轴PNG文件: {folder_path}")
    
    z_stack = []
    z_indices = []
    for idx, file_path in enumerate(z_files):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {file_path}")
        z_stack.append(img)
        z_indices.append(idx)
    
    return np.array(z_stack, dtype=np.uint16), np.array(z_indices), len(z_files)

def get_max_z_index(z_stack):
    max_indices = np.argmax(z_stack, axis=0)
    max_values = np.max(z_stack, axis=0)
    return max_indices, max_values

def apply_threshold_to_z(max_indices, max_values, threshold):
    mask = max_values < threshold
    filtered_indices = max_indices.copy()
    filtered_indices[mask] = -1
    return filtered_indices

def enhance_z_color_contrast(z_indices_data, total_z_layers, gamma=0.4):
    z_valid = z_indices_data >= 0
    z_normalized = np.zeros_like(z_indices_data, dtype=np.float32)
    
    z_float = z_indices_data[z_valid].astype(np.float32)
    z_min = 0.0
    z_max = float(total_z_layers - 1)
    
    if z_max > 0:
        z_norm = (z_float - z_min) / (z_max - z_min)
        z_enhanced = np.power(z_norm, gamma)
        z_normalized[z_valid] = z_enhanced * 255.0
    
    return z_normalized.astype(np.uint8)

def create_single_color_gradient(z_enhanced, z_indices_data):
    height, width = z_enhanced.shape
    colored_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    intensity = z_enhanced.astype(np.float32) / 255.0
    
    colored_map[:, :, 2] = (intensity * 255).astype(np.uint8)
    colored_map[:, :, 1] = (intensity * 100).astype(np.uint8)
    colored_map[:, :, 0] = (intensity * 50).astype(np.uint8)
    
    white_mask = z_indices_data == -1
    colored_map[white_mask] = [255, 255, 255]
    
    return colored_map

def main():
    input_folder = "data_simulation/images/87"
    output_depth_map = "outputs/simulation/fm_palette/inference_depth/max_depth_map.png"
    threshold_value = 120
    gamma_value = 0.4
    
    os.makedirs(os.path.dirname(output_depth_map), exist_ok=True)
    
    try:
        print("读取z轴图像栈...")
        z_stack, z_indices, total_layers = read_z_stack_images(input_folder)
        print(f"读取完成 - 共{total_layers}层, 图像尺寸: {z_stack.shape[1]}x{z_stack.shape[2]}")
        
        print("获取每个像素最大值对应的z层索引...")
        max_z_indices, max_pixel_values = get_max_z_index(z_stack)
        
        print("应用像素值阈值筛选...")
        filtered_z_indices = apply_threshold_to_z(max_z_indices, max_pixel_values, threshold_value)
        
        print("增强颜色对比度并生成深度图...")
        z_enhanced = enhance_z_color_contrast(filtered_z_indices, total_layers, gamma_value)
        depth_map = create_single_color_gradient(z_enhanced, filtered_z_indices)
        
        print("保存深度图...")
        cv2.imwrite(output_depth_map, depth_map, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        zero_pixels = np.sum(filtered_z_indices == -1)
        total_pixels = filtered_z_indices.size
        print("处理完成！")
        print(f"- 深度图已保存至: {output_depth_map}")
        print(f"- 使用阈值: {threshold_value}, 对比度增强Gamma值: {gamma_value}")
        print(f"- 白色像素数(阈值过滤): {zero_pixels}/{total_pixels} ({zero_pixels/total_pixels*100:.2f}%)")
        print(f"- z层索引范围: 0 ~ {total_layers-1}")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()