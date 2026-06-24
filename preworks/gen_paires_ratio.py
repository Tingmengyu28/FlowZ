#!/usr/bin/env python3
"""
生成deepZ图像对配对信息
遍历base_path中的所有子文件夹（每个文件夹为一个image stack），
按TRAIN_RATIO划分训练/验证集（无重叠stack），生成指定数量图像对配对信息
"""

import os
import random
import re
from pathlib import Path

def parse_z_index(filename):
    """从文件名中解析z索引，格式如 z1.png, z10.png"""
    match = re.search(r'z(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def collect_image_groups(base_path):
    """
    收集所有图像组信息
    :param base_path: 基础路径，子文件夹为各个image stack（序号1/2/3...）
    :return: 字典 {stack_name: [(z_index, file_path), ...]}  # stack_name为文件夹名（1/2/3...）
    """
    image_groups = {}
    base_path = Path(base_path)
    if not base_path.exists():
        return image_groups

    # 遍历base_path下所有**一级子文件夹**（每个文件夹是一个image stack）
    for stack_dir in base_path.iterdir():
        if not stack_dir.is_dir():
            continue
        stack_name = stack_dir.name  # stack序号：1/2/3...（原文件夹名）
        # 收集当前stack下所有png文件
        png_files = list(stack_dir.glob('*.png'))
        if not png_files:
            continue
        # 解析z索引并按序号排序
        z_images = []
        for png_file in png_files:
            z_index = parse_z_index(png_file.name)
            if z_index is not None:
                z_images.append((z_index, str(png_file)))
        z_images.sort(key=lambda x: x[0])  # 按z从小到大排序
        if len(z_images) >= 2:  # 至少2张图才能生成配对
            image_groups[stack_name] = z_images

    return image_groups

def generate_all_possible_pairs(channel_data, L_min, L_max):
    """
    生成所有可能的图像对（同stack内生成，避免跨stack）
    :param channel_data: 图像组数据 {stack_name: [(z_index, file_path), ...]}
    :param L_min/L_max: z索引最小/最大差值
    :return: 配对列表 [(image1_path, image2_path, z_diff), ...]
    """
    used_pairs = set()
    all_possible_pairs = []
    for stack_name, z_images in channel_data.items():
        if len(z_images) < 2:
            continue
        # 同stack内生成所有配对（i≠j，z差在指定范围）
        for i in range(len(z_images)):
            for j in range(len(z_images)):
                if i == j:
                    continue
                z1, path1 = z_images[i]
                z2, path2 = z_images[j]
                z_diff = z2 - z1
                # 过滤z差范围：±[L_min, L_max]
                if (L_min <= z_diff <= L_max) or (-L_max <= z_diff <= -L_min):
                    diff_value = z_diff / L_max  # 归一化z差
                    pair_key = (path1, path2)
                    if pair_key not in used_pairs:
                        all_possible_pairs.append((path1, path2, diff_value))
                        used_pairs.add(pair_key)
    return all_possible_pairs

def save_pairs_to_file(pairs, output_file):
    """将配对信息保存到文件，制表符分隔"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for img1_path, img2_path, z_diff in pairs:
            f.write(f"{img1_path}\t{img2_path}\t{z_diff}\n")
    print(f"已保存 {len(pairs)} 条配对 -> {output_file}")

def main():
    """主函数：自动统计stack、比例划分训练验证、生成配对"""
    # ===================== 可配置参数（核心！按需修改）=====================
    random.seed(42)  # 固定随机种子，结果可复现
    BASE_PATH = "data_simulation/images"  # 你的图像根路径（子文件夹为1/2/3...）
    OUTPUT_PATH = "data_simulation/pairs"         # 配对文件输出路径
    TRAIN_RATIO = 0.9                      # 训练集比例（0.8=80%训练，20%验证）
    L_MIN, L_MAX = 2, 12                   # z索引最小/最大差值
    TARGET_TRAIN_PAIRS = 150000            # 训练集目标配对数
    TARGET_VAL_PAIRS = 2000                # 验证集目标配对数
    # ======================================================================

    print("="*50)
    print("开始收集image stack信息...")
    # 1. 收集所有image stack（自动读取BASE_PATH下的子文件夹）
    image_groups = collect_image_groups(BASE_PATH)
    if not image_groups:
        print(f"错误：{BASE_PATH} 下未找到有效image stack（需含z*.png的文件夹）")
        return

    # 2. 统计基础信息
    total_stacks = len(image_groups)
    total_images = sum(len(imgs) for imgs in image_groups.values())
    all_stack_names = sorted(image_groups.keys(), key=lambda x: int(x))  # 按数字排序stack（1/2/3...）
    print(f"成功收集 -> 总stack数：{total_stacks} | 总图像数：{total_images}")
    print(f"所有stack序号：{all_stack_names}")

    # 3. 按比例划分训练/验证集（无重叠stack，随机划分）
    random.shuffle(all_stack_names)  # 随机打乱stack
    train_split = int(total_stacks * TRAIN_RATIO)
    train_stack_names = all_stack_names[:train_split]
    val_stack_names = all_stack_names[train_split:]
    # 分离训练/验证集的图像数据
    train_data = {s: image_groups[s] for s in train_stack_names}
    val_data = {s: image_groups[s] for s in val_stack_names}
    print("="*50)
    print(f"按比例划分 -> 训练集stack：{sorted(train_stack_names, key=int)}（共{len(train_stack_names)}个）")
    print(f"按比例划分 -> 验证集stack：{sorted(val_stack_names, key=int)}（共{len(val_stack_names)}个）")
    if not train_data:
        print("错误：训练集无有效stack！请降低TRAIN_RATIO或增加stack数量")
        return
    if not val_data:
        print("错误：验证集无有效stack！请提高TRAIN_RATIO或增加stack数量")
        return

    # 4. 生成训练/验证集所有可能的配对
    print("="*50)
    print("开始生成所有可能的图像对...")
    train_possible_pairs = generate_all_possible_pairs(train_data, L_MIN, L_MAX)
    val_possible_pairs = generate_all_possible_pairs(val_data, L_MIN, L_MAX)
    print(f"训练集可生成配对数：{len(train_possible_pairs)} | 目标数：{TARGET_TRAIN_PAIRS}")
    print(f"验证集可生成配对数：{len(val_possible_pairs)} | 目标数：{TARGET_VAL_PAIRS}")

    # 5. 按目标数筛选配对（不足则取全部，足够则随机采样）
    def sample_pairs(possible_pairs, target):
        if len(possible_pairs) <= target:
            return possible_pairs
        return random.sample(possible_pairs, target)
    train_pairs = sample_pairs(train_possible_pairs, TARGET_TRAIN_PAIRS)
    val_pairs = sample_pairs(val_possible_pairs, TARGET_VAL_PAIRS)

    # 6. 保存配对文件
    print("="*50)
    print("开始保存配对文件...")
    train_output_file = os.path.join(OUTPUT_PATH, "train_pairs.txt")
    val_output_file = os.path.join(OUTPUT_PATH, "val_pairs.txt")
    save_pairs_to_file(train_pairs, train_output_file)
    save_pairs_to_file(val_pairs, val_output_file)

    # 7. 生成汇总信息
    print("="*50)
    print("✅ 所有处理完成！汇总信息：")
    print(f"训练集配对数：{len(train_pairs)} | 保存路径：{train_output_file}")
    print(f"验证集配对数：{len(val_pairs)} | 保存路径：{val_output_file}")
    print(f"训练/验证集stack无重叠 | 划分比例：{TRAIN_RATIO*100}%/{(1-TRAIN_RATIO)*100}%")
    print("="*50)

if __name__ == "__main__":
    main()