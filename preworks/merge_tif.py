import os
import shutil
import tifffile
import numpy as np
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_single_file(tif_info):
    """
    处理单个TIF文件
    
    :param tif_info: 包含文件信息的字典
    :return: (tif_info, img, error_msg)
    """
    try:
        img = tifffile.imread(tif_info['full_path'])
        return tif_info, img, None
    except Exception as e:
        return tif_info, None, str(e)


def handle_directory_subprocess(subdir_path, output_dir):
    """
    处理单个目录的子进程函数
    
    :param subdir_path: 子目录路径
    :param output_dir: 输出目录路径
    :return: 处理结果信息
    """
    subdir = os.path.basename(subdir_path)
    print(f"Processing directory: {subdir_path}")
    
    # 创建对应子目录的输出文件夹
    group_output_dir = os.path.join(output_dir, subdir)
    os.makedirs(group_output_dir, exist_ok=True)
    
    # 查找所有TIF文件并解析其命名
    tif_files = []
    for file in os.listdir(subdir_path):
        if file.lower().endswith(('.tif', '.tiff')):
            # 解析文件名，查找_zXX_chYY格式
            match = re.search(r'_z(\d+)_ch(\d+)', file)
            if match:
                z_index = int(match.group(1))
                ch_index = int(match.group(2))
                tif_files.append({
                    'filename': file,
                    'z_index': z_index,
                    'ch_index': ch_index,
                    'full_path': os.path.join(subdir_path, file)
                })
    
    # 检查是否满足条件：恰好3个文件且每个都有多个z层（即每个TIF文件本身有多个切片）
    if len(tif_files) == 3:
        # 检查每个TIF文件是否包含多个z层（多维数组）
        files_with_multiple_z = 0
        for tif_info in tif_files:
            try:
                img = tifffile.imread(tif_info['full_path'])
                # 检查图像是否是多维的（有多个z层）
                if img.ndim >= 3:  # 至少是3维：z, height, width
                    files_with_multiple_z += 1
                elif img.ndim == 2:  # 2维是单个图像
                    pass  # 不增加计数
            except Exception as e:
                print(f"    Error reading {tif_info['filename']} to check dimensions: {e}")
        
        # 如果所有3个文件都有多个z层
        if files_with_multiple_z == 3:
            print(f"  Found exactly 3 files and all have multiple z layers in {subdir}, copying directly")
            
            # 直接复制文件到输出目录
            for tif_info in tif_files:
                output_filename = tif_info['filename']
                output_path = os.path.join(group_output_dir, output_filename)
                
                # 复制文件
                shutil.copy2(tif_info['full_path'], output_path)
                print(f"    Copied: {tif_info['filename']} to {output_path}")
                
            return f"Directory {subdir}: Copied {len(tif_files)} files directly to {group_output_dir}"
    
    # 检查是否满足条件：多个文件且符合命名格式（大于3个）
    if len(tif_files) > 3:  # 大于3个文件
        print(f"  Found {len(tif_files)} TIF files with z/ch format in {subdir}")
        
        # 按照channel分组
        channel_groups = defaultdict(list)
        for tif_info in tif_files:
            channel_groups[tif_info['ch_index']].append(tif_info)
        
        # 为每个channel创建多层TIF文件
        for ch_index in sorted(channel_groups.keys()):
            channel_files = channel_groups[ch_index]
            
            # 按z索引排序
            channel_files.sort(key=lambda x: x['z_index'])
            
            print(f"  Channel {ch_index}: Processing {len(channel_files)} slices")
            
            # 使用线程池并行读取图像数据
            image_stack = []
            with ThreadPoolExecutor(max_workers=min(len(channel_files), 8)) as executor:
                # 提交所有读取任务
                futures = [executor.submit(process_single_file, tif_info) for tif_info in channel_files]
                
                # 收集结果
                for future in as_completed(futures):
                    tif_info, img, error = future.result()
                    if error:
                        print(f"    Error reading {tif_info['filename']}: {error}")
                    else:
                        image_stack.append(img)
                        print(f"    Reading: {tif_info['filename']}, shape: {img.shape}")
            
            if image_stack:
                # 堆叠图像
                stacked_img = np.stack(image_stack, axis=0)  # shape: (z_slices, height, width)
                
                # 确保输出目录存在（已在上面创建）
                
                # 生成输出文件名
                output_filename = f"{subdir}_ch{ch_index:02d}.tif"
                output_path = os.path.join(group_output_dir, output_filename)
                
                # 保存多层TIF文件（移除compress参数）
                tifffile.imwrite(output_path, stacked_img)
                
                print(f"    Saved merged TIF for channel {ch_index}: {output_path}, shape: {stacked_img.shape}")
            else:
                print(f"    No valid images found for channel {ch_index}")
                
        return f"Directory {subdir}: Processed {len(tif_files)} files into {len(channel_groups)} channels in {group_output_dir}"
    else:
        msg = f"  Skipping {subdir} - doesn't meet criteria ({len(tif_files)} files)"
        print(msg)
        return msg


def merge_tif_files_by_channel(input_dir, output_dir, max_workers=4):
    """
    遍历输入目录，将具有相同channel但不同z层的TIF文件合并成一个多层TIF文件
    
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    :param max_workers: 最大工作线程数
    """
    # 获取所有子目录
    subdirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    
    print(f"Found {len(subdirs)} subdirectories to process")
    
    # 使用线程池并行处理各个目录
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有目录处理任务
        futures = [executor.submit(handle_directory_subprocess, subdir, output_dir) for subdir in subdirs]
        
        # 收集结果
        for future in as_completed(futures):
            result = future.result()
            print(result)


def main():
    # 定义输入和输出路径
    input_directory = "/data1/azt/cv/recoverZ/data/tif/CFC"
    output_directory = "/data1/azt/cv/recoverZ/data/tif/CFC_merged"
    
    print("Starting TIF merging process...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    # 执行并行合并操作
    merge_tif_files_by_channel(input_directory, output_directory)
    
    print("\nTIF merging process completed!")


if __name__ == "__main__":
    main()