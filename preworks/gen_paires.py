#!/usr/bin/env python3
"""
生成deepZ图像对配对信息
遍历inputs/microscopy_multiple/fixed_hela/deepZ中的所有文件，
生成大约20000行图像对配对信息。
"""

import os
import random
import re
from pathlib import Path

def parse_z_index(filename):
    """
    从文件名中解析z索引
    
    :param filename: 文件名，格式如 z1.png, z10.png
    :return: z索引数字，如果解析失败返回None
    """
    match = re.search(r'z(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def collect_image_groups(base_path):
    """
    收集所有图像组信息
    
    :param base_path: 基础路径 inputs/microscopy_multiple/fixed_hela/deepZ
    :return: 字典 {channel: {group: [(z_index, file_path), ...]}}
    """
    image_groups = {}
    base_path = Path(base_path)
    
    # 遍历所有通道
    for channel in ['ch0', 'ch1', 'ch2']:
        channel_path = base_path / channel
        if not channel_path.exists():
            continue
            
        image_groups[channel] = {}
        
        # 遍历所有组
        for group_num in range(1, 13):  # 01-12
            group_name = f"{group_num:02d}"
            group_path = channel_path / group_name
            
            if not group_path.exists():
                continue
                
            # 收集所有png文件
            png_files = list(group_path.glob('*.png'))
            if not png_files:
                continue
                
            # 解析z索引并排序
            z_images = []
            for png_file in png_files:
                z_index = parse_z_index(png_file.name)
                if z_index is not None:
                    z_images.append((z_index, str(png_file)))
            
            # 按z索引排序
            z_images.sort(key=lambda x: x[0])
            
            if len(z_images) >= 2:  # 至少需要2张图像才能形成配对
                image_groups[channel][group_name] = z_images
    
    return image_groups

def generate_all_possible_pairs(channel_data, L_min, L_max):
    """
    为单个通道生成所有可能的图像对配对信息
    
    :param channel_data: 通道的图像组数据 {group: [(z_index, file_path), ...]}
    :param L_min: 最小z索引差值
    :param L_max: 最大z索引差值
    :return: 所有可能的配对列表 [(image1_path, image2_path, z_diff), ...]
    """
    used_pairs = set()  # 用于避免重复
    all_possible_pairs = []
    
    for group_name, z_images in channel_data.items():
        if len(z_images) < 2:
            continue
            
        # 在同一组内生成所有可能的配对
        for i in range(len(z_images)):
            for j in range(len(z_images)):
                if i == j:
                    continue
                    
                z1, path1 = z_images[i]
                z2, path2 = z_images[j]
                
                # 计算z索引差距：第二个 - 第一个
                z_diff = z2 - z1
                
                # 检查z索引差距是否在 [L_min, L_max] 或 [-L_max, -L_min] 范围内
                if (L_min <= z_diff <= L_max) or (-L_max <= z_diff <= -L_min):
                    # 使用原始顺序：第一个图像 → 第二个图像
                    first_path, second_path = path1, path2
                    diff_value = z_diff / L_max
                    
                    pair_key = (first_path, second_path)
                    if pair_key not in used_pairs:
                        all_possible_pairs.append((first_path, second_path, diff_value))
                        used_pairs.add(pair_key)
    
    return all_possible_pairs

def generate_pairs_for_channel(channel_data, channel_name, target_pairs=20000, L_min=10, L_max=30):
    """
    为单个通道生成图像对配对信息（保留原函数用于兼容性）
    
    :param channel_data: 通道的图像组数据 {group: [(z_index, file_path), ...]}
    :param channel_name: 通道名称
    :param target_pairs: 目标配对数量
    :param L_min: 最小z索引差值
    :param L_max: 最大z索引差值
    :return: 配对列表 [(image1_path, image2_path, z_diff), ...]
    """
    all_possible_pairs = generate_all_possible_pairs(channel_data, L_min, L_max)
    
    print(f"{channel_name} 通道总共有 {len(all_possible_pairs)} 个可能的配对")
    
    # 随机选择配对
    if len(all_possible_pairs) < target_pairs:
        print(f"警告：{channel_name} 通道可能的配对数量({len(all_possible_pairs)})少于目标数量({target_pairs})")
        target_pairs = len(all_possible_pairs)
    
    # 随机选择不重复的配对
    selected_pairs = random.sample(all_possible_pairs, target_pairs)
    
    return selected_pairs

def save_pairs_to_file(pairs, output_file):
    """
    将配对信息保存到文件
    
    :param pairs: 配对列表
    :param output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for img1_path, img2_path, z_diff in pairs:
            f.write(f"{img1_path}\t{img2_path}\t{z_diff}\n")
    
    print(f"已生成 {len(pairs)} 行配对信息，保存到: {output_file}")

def main():
    """主函数"""
    # 设置随机种子以确保结果可重现
    random.seed(42)
    
    # 设置路径
    base_path = "data/images"
    output_path = "data/pairs"
    L_min, L_max = 5, 25
    train_target_pairs = 100000  # 训练集目标配对数量
    val_target_pairs = 2000     # 验证集目标配对数量
    val_groups = ["10"]  # 可以根据需要扩展
    
    # 创建输出目录结构
    train_dir = os.path.join(output_path, "train")
    val_dir = os.path.join(output_path, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print("开始收集图像信息...")
    
    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        print(f"错误：基础路径不存在: {base_path}")
        return
    
    # 收集图像组信息
    image_groups = collect_image_groups(base_path)
    
    if not image_groups:
        print("错误：未找到任何图像文件")
        return
    
    # 统计信息
    total_channels = len(image_groups)
    total_groups = sum(len(groups) for groups in image_groups.values())
    total_images = sum(len(images) for groups in image_groups.values() for images in groups.values())
    
    print(f"  - 通道数: {total_channels}")
    print(f"  - 组数: {total_groups}")
    print(f"  - 总图像数: {total_images}")
    
    # 为每个通道生成配对
    print("\n开始为每个通道生成图像对配对...")
    
    total_pairs_generated = 0
    
    for channel in ['ch0', 'ch1', 'ch2']:
        if channel not in image_groups:
            print(f"警告：未找到 {channel} 通道数据")
            continue
            
        print(f"\n处理 {channel} 通道...")
        channel_data = image_groups[channel]
        
        # 统计该通道的图像数量
        total_images_in_channel = sum(len(images) for images in channel_data.values())
        print(f"{channel} 通道总共有 {total_images_in_channel} 张图像，分布在 {len(channel_data)} 个组中")
        
        # 获取所有组名
        all_groups = list(channel_data.keys())
        
        # 确保验证组在实际存在的组中
        val_groups = [group for group in val_groups if group in all_groups]
        
        # 训练组是除了验证组之外的所有组
        train_groups = [group for group in all_groups if group not in val_groups]
        
        print(f"  - 训练组: {train_groups}")
        print(f"  - 验证组: {val_groups}")
        
        # 为训练组生成所有可能的配对
        train_channel_data = {group: images for group, images in channel_data.items() if group in train_groups}
        train_possible_pairs = generate_all_possible_pairs(train_channel_data, L_min, L_max)
        
        # 为验证组生成所有可能的配对
        val_channel_data = {group: images for group, images in channel_data.items() if group in val_groups}
        val_possible_pairs = generate_all_possible_pairs(val_channel_data, L_min, L_max)
        
        # 检查训练集配对数量
        if len(train_possible_pairs) < train_target_pairs:
            print(f"警告：{channel} 通道训练组可能的配对数量({len(train_possible_pairs)})少于目标数量({train_target_pairs})")
            train_actual_pairs = len(train_possible_pairs)
        else:
            train_actual_pairs = train_target_pairs
        
        # 检查验证集配对数量
        if len(val_possible_pairs) < val_target_pairs:
            print(f"警告：{channel} 通道验证组可能的配对数量({len(val_possible_pairs)})少于目标数量({val_target_pairs})")
            val_actual_pairs = len(val_possible_pairs)
        else:
            val_actual_pairs = val_target_pairs
        
        # 随机打乱训练集配对
        random.shuffle(train_possible_pairs)
        # 随机打乱验证集配对
        random.shuffle(val_possible_pairs)
        
        # 选择目标数量的配对
        train_pairs = train_possible_pairs[:train_actual_pairs]
        val_pairs = val_possible_pairs[:val_actual_pairs]
        
        # 保存训练集
        train_file = os.path.join(train_dir, f"{channel}_train_pairs.txt")
        save_pairs_to_file(train_pairs, train_file)
        
        # 保存验证集
        val_file = os.path.join(val_dir, f"{channel}_val_pairs.txt")
        save_pairs_to_file(val_pairs, val_file)
        
        total_pairs_generated += len(train_pairs) + len(val_pairs)
        
        # 显示该通道的示例
        print(f"\n{channel} 通道训练集前3个配对示例:")
        for i, (img1, img2, diff) in enumerate(train_pairs[:3]):
            # 提取组名
            group1 = os.path.basename(os.path.dirname(img1))
            group2 = os.path.basename(os.path.dirname(img2))
            print(f"  训练 {i+1}. {os.path.basename(img1)} (组{group1}) -> {os.path.basename(img2)} (组{group2}) (diff: {diff})")
            
        print(f"\n{channel} 通道验证集前3个配对示例:")
        for i, (img1, img2, diff) in enumerate(val_pairs[:3]):
            # 提取组名
            group1 = os.path.basename(os.path.dirname(img1))
            group2 = os.path.basename(os.path.dirname(img2))
            print(f"  验证 {i+1}. {os.path.basename(img1)} (组{group1}) -> {os.path.basename(img2)} (组{group2}) (diff: {diff})")
    
    print(f"总共生成了 {total_pairs_generated} 个配对")
            
    # 生成汇总信息
    print("\n生成的文件列表:")
    for channel in ['ch0', 'ch1', 'ch2']:
        # 检查训练集文件
        train_file = os.path.join(train_dir, f"{channel}_train_pairs.txt")
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                train_lines = len(f.readlines())
            print(f"  - 训练集: {train_file} ({train_lines} 行)")
        
        # 检查验证集文件
        val_file = os.path.join(val_dir, f"{channel}_val_pairs.txt")
        if os.path.exists(val_file):
            with open(val_file, 'r') as f:
                val_lines = len(f.readlines())
            print(f"  - 验证集: {val_file} ({val_lines} 行)")

if __name__ == "__main__":
    main()