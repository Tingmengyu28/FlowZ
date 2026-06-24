#!/usr/bin/env python3
"""
生成deepZ图像对配对信息（新增配对数量上限 + 修正分组粒度 + 新增z索引范围过滤）
遍历INPUT_DIR下的s1/s2子文件夹，以「split+group」为粒度划分训练/测试集，
限制训练/测试集配对数量上限（超过则随机筛选），且仅选取z索引在[S_min, S_max]范围内的图片。
"""

import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

def parse_z_index(filename: str) -> int | None:
    """
    从文件名中解析z索引（兼容z1.png/z10.png等格式）
    
    :param filename: 文件名（如 z1.png）
    :return: z索引数字，解析失败返回None
    """
    match = re.search(r'z(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def collect_image_groups(
    base_path: str | Path, 
    S_min: int,  # 新增：z索引最小值（包含）
    S_max: int   # 新增：z索引最大值（包含）
) -> Tuple[Dict[str, List[Tuple[int, str]]], List[str]]:
    """
    收集所有图像组信息（核心：生成唯一group_id = split_group，如s1_a01）
    新增：仅保留z索引在[S_min, S_max]范围内的图片
    
    :param base_path: 数据集根路径（INPUT_DIR）
    :param S_min: z索引最小值（包含）
    :param S_max: z索引最大值（包含）
    :return: 
        - image_groups: {group_id: [(z_index, file_path), ...]}（group_id=split_group）
        - all_group_ids: 所有唯一的group_id列表（如[s1_a01, s1_a02, s2_a01]）
    """
    image_groups = {}
    all_group_ids = []
    base_path = Path(base_path)
    
    # 新增：打印z索引过滤范围
    print(f"===== 开始过滤z索引：仅保留 [{S_min}, {S_max}] 范围内的切片 =====")
    
    # 遍历s1/s2（split层级）
    for split in ['s1', 's2']:
        split_path = base_path / split
        if not split_path.exists():
            print(f"警告：split路径不存在，跳过: {split_path}")
            continue
            
        # 遍历split下所有子文件夹（a01/a02/b01等，group层级）
        for group_path in split_path.iterdir():
            if not group_path.is_dir():
                continue
            
            # 生成唯一group_id（如s1_a01、s2_a01）
            group_name = group_path.name
            group_id = f"{split}_{group_name}"  # 核心修正：split+group作为唯一标识
            
            # 收集该group下所有png文件
            png_files = list(group_path.glob('*.png'))
            if not png_files:
                continue
                
            # 解析z索引并过滤：仅保留z∈[S_min, S_max]且解析有效的文件
            z_images = []
            for png_file in png_files:
                z_index = parse_z_index(png_file.name)
                # 新增：过滤z索引范围
                if z_index is not None and S_min <= z_index <= S_max:
                    z_images.append((z_index, str(png_file.resolve())))  # 保存绝对路径
            
            # 打印该group的过滤结果（调试用）
            print(f"Group {group_id}: 原始PNG数={len(png_files)}, 有效z索引数={len(z_images)} (z∈[{S_min},{S_max}])")
            
            # 按z索引排序（保证切片顺序）
            z_images.sort(key=lambda x: x[0])
            
            # 至少2张有效切片才纳入有效组
            if len(z_images) >= 2:
                image_groups[group_id] = z_images
                all_group_ids.append(group_id)
            else:
                print(f"警告：Group {group_id} 有效切片数不足2张，跳过")
    
    # 检查是否有有效数据
    if not image_groups:
        raise ValueError(f"错误：未找到任何有效图像组（需每个group至少2张z∈[{S_min},{S_max}]的切片）")
    
    print("\n===== 过滤完成 =====")
    print(f"有效group数（split+group）: {len(all_group_ids)}")
    print(f"示例group_id: {all_group_ids[:5]}")  # 打印前5个示例
    return image_groups, all_group_ids

def generate_all_possible_pairs(
    image_groups: Dict[str, List[Tuple[int, str]]],
    target_group_ids: List[str],
    L_min: int = 5,
    L_max: int = 25
) -> List[Tuple[str, str, float]]:
    """
    为指定group_id生成所有符合条件的图像配对（仅同一group_id内生成配对）
    注：因上游已过滤z范围，此处仅处理过滤后的有效切片
    
    :param image_groups: 完整图像组数据 {group_id: [(z_index, path), ...]}
    :param target_group_ids: 要生成配对的group_id列表（train/test）
    :param L_min: 最小z差值（绝对值）
    :param L_max: 最大z差值（绝对值）
    :return: 配对列表 [(img1_path, img2_path, normalized_z_diff), ...]
    """
    used_pairs = set()  # 避免重复配对（img1→img2 和 img2→img1 视为不同配对）
    valid_pairs = []
    
    # 遍历目标group_id
    for group_id in target_group_ids:
        if group_id not in image_groups:
            continue
        z_images = image_groups[group_id]
        
        # 同一group_id内生成配对（保证Z轴连续性）
        for i in range(len(z_images)):
            z1, path1 = z_images[i]
            for j in range(len(z_images)):
                if i == j:  # 跳过自身配对
                    continue
                    
                z2, path2 = z_images[j]
                z_diff = z2 - z1  # 计算z差值（img2 - img1）
                
                # 过滤差值范围：|z_diff| ∈ [L_min, L_max]
                if L_min <= abs(z_diff) <= L_max:
                    normalized_diff = z_diff / L_max  # 归一化到[-1,1]
                    pair_key = (path1, path2)
                    
                    # 避免重复添加同一配对
                    if pair_key not in used_pairs:
                        valid_pairs.append((path1, path2, normalized_diff))
                        used_pairs.add(pair_key)
    
    return valid_pairs

def split_train_test_groups(
    all_group_ids: List[str],
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    按比例划分训练/测试group_id（以split+group为粒度，保证无重叠）
    
    :param all_group_ids: 所有唯一的group_id列表（如[s1_a01, s2_a01]）
    :param train_ratio: 训练集占比
    :param random_seed: 随机种子（保证可重现）
    :return: (train_group_ids, test_group_ids)
    """
    random.seed(random_seed)
    shuffled_group_ids = random.sample(all_group_ids, len(all_group_ids))  # 随机打乱
    
    # 划分训练/测试边界
    split_idx = int(len(shuffled_group_ids) * train_ratio)
    train_group_ids = shuffled_group_ids[:split_idx]
    test_group_ids = shuffled_group_ids[split_idx:]
    
    print(f"总group_id数: {len(all_group_ids)}")
    print(f"训练集group_id数: {len(train_group_ids)} ({train_ratio*100}%)")
    print(f"测试集group_id数: {len(test_group_ids)} ({(1-train_ratio)*100}%)")
    print(f"训练集group_id示例: {sorted(train_group_ids)[:5]}")  # 前5个示例
    print(f"测试集group_id示例: {sorted(test_group_ids)[:5]}")  # 前5个示例
    
    return train_group_ids, test_group_ids

def filter_pairs_by_max_limit(pairs: List[Tuple[str, str, float]], max_pairs: int, set_name: str, random_seed: int = 42) -> List[Tuple[str, str, float]]:
    """
    按上限过滤配对数量：超过上限则随机筛选，否则保留全部
    
    :param pairs: 原始配对列表
    :param max_pairs: 最大配对数量上限
    :param set_name: 数据集名称（train/test），用于打印日志
    :param random_seed: 随机种子（保证可重现）
    :return: 过滤后的配对列表
    """
    random.seed(random_seed)  # 固定种子保证可重现
    original_count = len(pairs)
    
    if original_count <= max_pairs:
        print(f"{set_name}集原始配对数({original_count}) ≤ 上限({max_pairs})，保留全部")
        return pairs
    else:
        # 随机筛选到上限数量
        filtered_pairs = random.sample(pairs, max_pairs)
        print(f"{set_name}集原始配对数({original_count}) > 上限({max_pairs})，随机筛选到{max_pairs}条")
        return filtered_pairs

def save_pairs_to_file(pairs: List[Tuple[str, str, float]], output_file: str | Path):
    """
    将配对信息保存到txt文件（制表符分隔）
    
    :param pairs: 配对列表
    :param output_file: 输出文件路径
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for img1, img2, diff in pairs:
            f.write(f"{img1}\t{img2}\t{diff:.6f}\n")  # 保留6位小数
    
    print(f"\n已保存 {len(pairs)} 条配对到: {output_file}")

def main():
    """主函数：配置参数 + 执行流程"""
    # ===================== 可配置参数 =====================
    INPUT_DIR = "data_BBBC006/images"          # 数据集根路径（s1/s2所在目录）
    OUTPUT_DIR = "data_BBBC006/pairs"          # 配对文件输出目录
    L_MIN = 5                          # 最小z差值（绝对值）
    L_MAX = 15                         # 最大z差值（绝对值）
    TRAIN_RATIO = 0.9                  # 训练集占比（group_id划分）
    RANDOM_SEED = 42                   # 随机种子（保证结果可重现）
    TRAIN_MAX_PAIRS = 120000            # 训练集配对数量上限
    TEST_MAX_PAIRS = 4000              # 测试集配对数量上限
    S_MIN = 5                          # 新增：z索引最小值（包含）
    S_MAX = 25                         # 新增：z索引最大值（包含）
    # ======================================================

    # 新增：校验S_MIN和S_MAX的合法性
    if S_MIN > S_MAX:
        raise ValueError(f"错误：S_MIN({S_MIN}) 不能大于 S_MAX({S_MAX})")
    if S_MIN < 0:
        raise ValueError(f"错误：S_MIN({S_MIN}) 不能为负数")

    # 1. 收集图像组信息（新增传递S_MIN/S_MAX参数）
    print("===== 开始收集图像组信息 =====")
    try:
        image_groups, all_group_ids = collect_image_groups(INPUT_DIR, S_MIN, S_MAX)
    except ValueError as e:
        print(e)
        return
    
    # 2. 划分训练/测试group_id（以split+group为粒度）
    train_group_ids, test_group_ids = split_train_test_groups(
        all_group_ids, 
        train_ratio=TRAIN_RATIO,
        random_seed=RANDOM_SEED
    )
    
    # 3. 生成训练/测试配对
    print("\n===== 生成训练集配对 =====")
    train_pairs_original = generate_all_possible_pairs(
        image_groups, 
        target_group_ids=train_group_ids,
        L_min=L_MIN,
        L_max=L_MAX
    )
    # 按上限过滤训练集配对
    train_pairs = filter_pairs_by_max_limit(
        train_pairs_original, 
        max_pairs=TRAIN_MAX_PAIRS,
        set_name="训练",
        random_seed=RANDOM_SEED
    )
    
    print("\n===== 生成测试集配对 =====")
    test_pairs_original = generate_all_possible_pairs(
        image_groups, 
        target_group_ids=test_group_ids,
        L_min=L_MIN,
        L_max=L_MAX
    )
    # 按上限过滤测试集配对
    test_pairs = filter_pairs_by_max_limit(
        test_pairs_original, 
        max_pairs=TEST_MAX_PAIRS,
        set_name="测试",
        random_seed=RANDOM_SEED
    )
    
    # 4. 保存配对文件
    train_output_file = os.path.join(OUTPUT_DIR, "train_pairs.txt")
    test_output_file = os.path.join(OUTPUT_DIR, "test_pairs.txt")
    
    save_pairs_to_file(train_pairs, train_output_file)
    save_pairs_to_file(test_pairs, test_output_file)
    
    # 5. 输出汇总信息（新增z范围说明）
    print("\n===== 最终汇总 =====")
    print(f"训练集配对文件: {train_output_file} ({len(train_pairs)} 行)")
    print(f"测试集配对文件: {test_output_file} ({len(test_pairs)} 行)")
    print(f"总配对数: {len(train_pairs) + len(test_pairs)}")
    print(f"配置参数：L_MIN={L_MIN}, L_MAX={L_MAX}, 训练集上限={TRAIN_MAX_PAIRS}, 测试集上限={TEST_MAX_PAIRS}")
    print(f"Z索引过滤范围：[{S_MIN}, {S_MAX}]")  # 新增：打印z范围配置
    
    # 输出前3条配对示例
    if train_pairs:
        print("\n===== 训练集配对示例 =====")
        for i, (img1, img2, diff) in enumerate(train_pairs[:3]):
            # 提取group_id（split+group）
            group_id = os.path.basename(os.path.dirname(img1))
            split = os.path.basename(os.path.dirname(os.path.dirname(img1)))
            full_group_id = f"{split}_{group_id}"
            print(f"示例{i+1}: {full_group_id}/{os.path.basename(img1)} → {os.path.basename(img2)} (归一化差值: {diff:.6f})")
    
    if test_pairs:
        print("\n===== 测试集配对示例 =====")
        for i, (img1, img2, diff) in enumerate(test_pairs[:3]):
            group_id = os.path.basename(os.path.dirname(img1))
            split = os.path.basename(os.path.dirname(os.path.dirname(img1)))
            full_group_id = f"{split}_{group_id}"
            print(f"示例{i+1}: {full_group_id}/{os.path.basename(img1)} → {os.path.basename(img2)} (归一化差值: {diff:.6f})")

if __name__ == "__main__":
    main()