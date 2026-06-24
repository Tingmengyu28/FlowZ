# 导入必要的库
import os
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


def apply_lowpass_filter(image, sigma=1.0):
    """
    对图像应用低通滤波器（高斯滤波）以去除噪声
    
    :param image: 输入图像
    :param sigma: 高斯核的标准差，控制滤波强度
    :return: 滤波后的图像
    """
    # 应用高斯滤波
    filtered_image = ndimage.gaussian_filter(image, sigma=sigma)
    return filtered_image

def get_sorted_subdirs(input_folder):
    """
    获取并排序输入文件夹中的所有子文件夹
    
    :param input_folder: 输入文件夹路径
    :return: 排序后的子文件夹列表
    """
    subdirs = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    subdirs.sort()  # 按名称排序
    return subdirs

def extract_folder_number(folder_name, folder_counter):
    """
    从文件夹名称中提取编号，如果无法提取则使用计数器
    
    :param folder_name: 文件夹名称
    :param folder_counter: 文件夹计数器
    :return: 文件夹编号
    """
    match = re.search(r'(?:Con|WF)-(\d+)', folder_name)
    if match:
        return match.group(1)
    else:
        return str(folder_counter)

def save_image(image_data, output_path):
    """
    通用图像保存函数，处理不同数据类型的归一化
    
    :param image_data: 图像数据
    :param output_path: 输出路径
    """
    # 创建RGB图像（单通道转灰度）
    if image_data.ndim == 2:
        height, width = image_data.shape
        rgb_image = np.zeros((height, width, 3), dtype=image_data.dtype)
        rgb_image[:, :, 0] = image_data
        rgb_image[:, :, 1] = image_data
        rgb_image[:, :, 2] = image_data
    else:
        rgb_image = image_data
    
    rgb_normalized = rgb_image.astype(np.float32) * 255.0
    rgb_uint8 = rgb_normalized.astype(np.uint8)
    
    plt.imsave(output_path, rgb_uint8)

def process_and_save_image(img, idx, output_dir, sigma=1.0, prefix=""):
    """
    处理并保存单个图像
    
    :param img: 图像数据
    :param idx: 图像索引
    :param output_dir: 输出目录
    :param sigma: 滤波器参数
    :param prefix: 名称前缀
    :return: (output_path, idx) 或 None
    """
    # 应用高斯滤波去除噪声
    img_filtered = apply_lowpass_filter(img, sigma=sigma)
    
    # 命名格式: z1-zN.png
    output_filename = f"z{idx+1}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # 使用通用保存函数保存图像
    save_image(img_filtered, output_path)
    
    return output_path, idx

def process_single_tiff_file(file_path, idx, output_dir, sigma=1.0, prefix=""):
    """
    处理单个TIFF文件
    
    :param file_path: TIFF文件路径
    :param idx: 文件索引
    :param output_dir: 输出目录
    :param sigma: 滤波器参数
    :param prefix: 名称前缀
    :return: (output_path, idx) 或 None
    """
    # 读取TIFF图像
    img = tifffile.imread(file_path)
    
    # 处理多页TIFF，只取第一页
    if img.ndim > 2:
        img = img[0]
    
    # 应用高斯滤波去除噪声
    img = apply_lowpass_filter(img, sigma=sigma)
    
    # 命名格式: z1-zN.png
    output_filename = f"z{idx+1}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # 使用通用保存函数保存图像
    save_image(img, output_path)
    
    return output_path, idx

def process_multichannel_file(file_path, ch_idx, output_dir, sigma=1.0):
    """
    处理多通道单文件TIFF（包含多个页面）
    
    :param file_path: TIFF文件路径
    :param ch_idx: 通道索引
    :param output_dir: 输出目录
    :param sigma: 滤波器参数
    :return: [(ch_idx, output_path, idx), ...] 列表，包含所有slice的结果
    """
    results = []
    try:
        # 读取TIFF文件的所有页面
        with tifffile.TiffFile(file_path) as tif:
            # 首先获取所有页面的数据以确定最大值
            all_pages = [page.asarray() for page in tif.pages]
            
            # 计算整个TIF文件的最大值
            max_val = max(page.max() for page in all_pages)
            
            # 使用最大值对所有页面进行归一化并保存
            for idx, img in enumerate(all_pages):
                # 对每个slice进行最大值归一化
                if max_val > 0:
                    img = img.astype(np.float32) / max_val
                else:
                    img = img.astype(np.float32)
                
                # 应用高斯滤波去除噪声
                img = apply_lowpass_filter(img, sigma=sigma)
                
                # 命名格式: z1-zN.png
                output_filename = f"z{idx+1}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # 使用通用保存函数保存图像
                save_image(img, output_path)
                
                results.append((ch_idx, output_path, idx))
                
        return results
    except Exception as e:
        print(f"读取TIFF文件时出错 (通道 {ch_idx}): {e}")
        return results

def process_single_multichannel_tiff_file(file_path, idx, ch_idx, output_dir, sigma=1.0):
    """
    处理单通道多文件TIFF
    
    :param file_path: TIFF文件路径
    :param idx: 文件索引
    :param ch_idx: 通道索引
    :param output_dir: 输出目录
    :param sigma: 滤波器参数
    :return: (ch_idx, output_path, idx) 或 None
    """
    try:
        # 读取TIFF图像
        img = tifffile.imread(file_path)
        
        # 处理多页TIFF，只取第一页
        if img.ndim > 2:
            img = img[0]
        
        # 应用高斯滤波去除噪声
        img = apply_lowpass_filter(img, sigma=sigma)
        
        # 命名格式: z1-zN.png
        output_filename = f"z{idx+1}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 使用通用保存函数保存图像
        save_image(img, output_path)
        
        return ch_idx, output_path, idx
    except Exception as e:
        print(f"读取TIFF文件时出错 (通道 {ch_idx}): {e}")
    return None

def process_defocus_folder(input_folder, output_folder, num_ch=0):
    """
    处理defocus图像文件夹
    
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param num_ch: 通道数量，默认为0表示单通道
    """
    if num_ch <= 0:
        # 单通道处理模式
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有子文件夹并按名称排序，确保处理顺序一致
        subdirs = get_sorted_subdirs(input_folder)
        
        # 遍历排序后的子文件夹
        for folder_counter, folder_name in enumerate(subdirs, 1):
            root = os.path.join(input_folder, folder_name)
            files = sorted([f for f in os.listdir(root) if f.endswith('.tif') or f.endswith('.tiff')])
            
            if files:
                print(f"处理Defocus文件夹: {root}")
                print(f"总共找到 {len(files)} 个z位置")
                
                # 检查文件结构并决定读取方法
                use_new_method = False
                if files:
                    # 检查是否只有一个TIFF文件且包含多个页面
                    if len(files) == 1:
                        file_path = os.path.join(root, files[0])
                        try:
                            with tifffile.TiffFile(file_path) as tif:
                                if len(tif.pages) > 1:
                                    use_new_method = True
                                    print(f"使用新方法读取: 单个TIFF文件包含{len(tif.pages)}个页面")
                        except Exception as e:
                            print(f"读取TIFF文件时出错: {e}")
                
                # 提取文件夹名称用于命名
                folder_number = extract_folder_number(folder_name, folder_counter)
                
                # 为当前文件夹创建子文件夹
                folder_output_path = os.path.join(output_folder, folder_number)
                os.makedirs(folder_output_path, exist_ok=True)
                
                if use_new_method:
                    # 使用新方法读取单个TIFF文件的多个页面
                    file_path = os.path.join(root, files[0])
                    try:
                        with tifffile.TiffFile(file_path) as tif:
                            pages = [page.asarray() for page in tif.pages]
                            
                            # 计算整个TIF文件的最大值
                            max_val = max(page.max() for page in pages)
                            
                            # 使用最大值对所有页面进行归一化并保存
                            for idx, img in enumerate(pages):
                                # 对每个slice进行最大值归一化
                                if max_val > 0:
                                    img = img.astype(np.float32) / max_val
                                else:
                                    img = img.astype(np.float32)
                                
                                # 使用ThreadPoolExecutor并行处理所有页面
                                with ThreadPoolExecutor() as executor:
                                    # 提交所有任务
                                    futures = []
                                    future = executor.submit(
                                        process_and_save_image,
                                        img, idx, folder_output_path, sigma=1.0
                                    )
                                    futures.append(future)
                                
                                # 等待所有任务完成
                                for future in as_completed(futures):
                                    result = future.result()
                                    if result:
                                        output_path, idx = result
                                        print(f"已保存Defocus图像: {output_path} (原始索引: {idx})")
                    except Exception as e:
                        print(f"读取TIFF文件时出错: {e}")
                else:
                    continue
            else:
                print(f"在文件夹中未找到TIFF文件: {root}")
    else:
        # 多通道处理模式
        # 为不同通道创建不同的输出目录
        output_folders = []
        for i in range(num_ch):
            ch_output_folder = os.path.join(output_folder, f"ch{i}")
            os.makedirs(ch_output_folder, exist_ok=True)
            output_folders.append(ch_output_folder)
        
        # 获取所有子文件夹并按名称排序，确保处理顺序一致
        subdirs = get_sorted_subdirs(input_folder)
        
        # 遍历排序后的子文件夹
        for folder_counter, folder_name in enumerate(subdirs, 1):
            root = os.path.join(input_folder, folder_name)
            files = os.listdir(root)
            
            # 分离不同通道的文件并排序
            ch_files = []
            for i in range(num_ch):
                ch_files.append(sorted([f for f in files if f.endswith(f'_ch{i:02d}.tif') or f.endswith(f'_ch{i:02d}.tiff')]))
            
            # 修改条件检查，确保至少有一个通道有文件
            if any(ch_files):
                print(f"处理Defocus文件夹: {root}")
                # 获取总文件数（以第一个非空通道为准）
                total_files = 0
                for ch_file_list in ch_files:
                    if ch_file_list:
                        total_files = len(ch_file_list)
                        break
                print(f"总共找到 {total_files} 个z位置")
                
                # 检查文件结构并决定读取方法
                # 先检查文件夹中的文件数量是否等于num_ch
                use_new_method = len(ch_files[0]) == 1
                
                if use_new_method:
                    print("使用新方法读取: 每个通道一个TIFF文件，包含多个页面")
                else:
                    continue
                
                # 提取文件夹名称用于命名
                folder_number = extract_folder_number(folder_name, folder_counter)
                
                # 为当前文件夹创建子文件夹
                folder_output_paths = []
                for i in range(num_ch):
                    ch_folder_path = os.path.join(output_folders[i], folder_number)
                    os.makedirs(ch_folder_path, exist_ok=True)
                    folder_output_paths.append(ch_folder_path)
                
                with ThreadPoolExecutor(max_workers=num_ch) as executor:
                    futures = []
                    
                    for i in range(num_ch):
                        # 检查此通道是否有文件
                        if ch_files[i]:
                            file_path = os.path.join(root, ch_files[i][0])  # 每个通道只有一个文件
                            
                            future = executor.submit(
                                process_multichannel_file,
                                file_path, i, folder_output_paths[i], sigma=2.0
                            )
                            futures.append(future)
                    
                    # 等待所有任务完成
                    for future in as_completed(futures):
                        results = future.result()
                        if results:
                            for ch_idx, output_path, idx in results:
                                print(f"已保存ch{ch_idx:02d} Defocus图像: {output_path} (原始索引: {idx})")
            else:
                print(f"在文件夹中未找到TIFF文件: {root}")

def main():
    num_ch = 3

    # 定义输入和输出路径
    base_input_path = "data/tif/WF_merged"  # 修复了路径，添加了缺失的斜杠
    output_base_path = "data/images"
    
    print("开始处理Defocus图像...")
    process_defocus_folder(base_input_path, output_base_path, num_ch)
    
    print("\n所有图像处理完成!")

if __name__ == "__main__":
    main()