#!/usr/bin/env python3
"""
生成deepZ图像对配对信息（新增配对数量上限 + 修正分组粒度）
遍历INPUT_DIR下的所有子文件夹，以「group」为粒度划分训练/测试集，
限制训练/测试集配对数量上限（超过则随机筛选）
"""

import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

def parse_z_index(filename: str) -> Optional[int]:
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
    base_path: Union[str, Path]
) -> Tuple[Dict[str, List[Tuple[int, str]]], List[str], Dict[str, List[str]]]:
    """
    收集所有图像组信息。

    支持两种目录结构（自动探测）：
      新结构: group_name/patch_{id}/zX.png   ← pair 在 patch 内部生成
      旧结构: group_name/zX.png              ← pair 在 group 内部生成

    :param base_path: 数据集根路径（INPUT_DIR）
    :return:
        - image_groups: {compound_id: [(z_index, file_path), ...]}
          compound_id = "group_name"（旧结构）或 "group_name/patch_id"（新结构）
        - all_group_names: 所有唯一的 group_name 列表（用于 train/test 按 group 粒度划分）
        - group_to_patches: {group_name: [compound_id, ...]} 映射
    """
    image_groups: Dict[str, List[Tuple[int, str]]] = {}
    all_group_names: List[str] = []
    group_to_patches: Dict[str, List[str]] = {}
    base_path = Path(base_path)

    for group_path in sorted(base_path.iterdir()):
        if not group_path.is_dir():
            continue

        group_name = group_path.name

        # 探测是否有 patch 子目录（新结构）
        patch_dirs = sorted([d for d in group_path.iterdir() if d.is_dir()])

        compound_ids = []

        if patch_dirs:
            # 新结构: group_name / patch_{id} / zX.png
            for patch_dir in patch_dirs:
                patch_id = patch_dir.name
                compound_id = f"{group_name}/{patch_id}"

                png_files = sorted(patch_dir.glob('*.png'))
                z_images = []
                for f in png_files:
                    z_idx = parse_z_index(f.name)
                    if z_idx is not None:
                        z_images.append((z_idx, str(f.resolve())))
                z_images.sort(key=lambda x: x[0])

                if len(z_images) >= 2:
                    image_groups[compound_id] = z_images
                    compound_ids.append(compound_id)
                else:
                    print(f"  跳过 {compound_id}: 有效切片数 {len(z_images)} < 2")
        else:
            # 旧结构: group_name / zX.png
            png_files = sorted(group_path.glob('*.png'))
            z_images = []
            for f in png_files:
                z_idx = parse_z_index(f.name)
                if z_idx is not None:
                    z_images.append((z_idx, str(f.resolve())))
            z_images.sort(key=lambda x: x[0])

            if len(z_images) >= 2:
                image_groups[group_name] = z_images
                compound_ids.append(group_name)
            else:
                print(f"  跳过 {group_name}: 有效切片数 {len(z_images)} < 2")

        if compound_ids:
            all_group_names.append(group_name)
            group_to_patches[group_name] = compound_ids
            print(f"Group {group_name}: {len(compound_ids)} 个有效块")

    if not image_groups:
        raise ValueError("错误：未找到任何有效图像组（每个块至少需要 2 张切片）")

    print("\n===== 过滤完成 =====")
    print(f"有效 group 数: {len(all_group_names)}")
    print(f"有效块总数: {len(image_groups)}")
    print(f"示例 group: {all_group_names[:5]}")
    return image_groups, all_group_names, group_to_patches

def generate_all_possible_pairs(
    image_groups: Dict[str, List[Tuple[int, str]]],
    target_compound_ids: List[str],
    L_min: int = 5,
    L_max: int = 25
) -> List[Tuple[str, str, float]]:
    """
    为指定 compound_id 生成所有符合条件的图像配对。
    仅在同一 compound_id 内（同一 group 的同一 patch）生成。

    :param image_groups: 完整图像组数据 {compound_id: [(z_index, path), ...]}
    :param target_compound_ids: 要生成配对的 compound_id 列表
    :param L_min: 最小 z 差值（绝对值）
    :param L_max: 最大 z 差值（绝对值）
    :return: 配对列表 [(img1_path, img2_path, normalized_z_diff), ...]
    """
    used_pairs: set = set()
    valid_pairs: list = []

    for compound_id in target_compound_ids:
        if compound_id not in image_groups:
            continue
        z_images = image_groups[compound_id]

        for i in range(len(z_images)):
            z1, path1 = z_images[i]
            for j in range(len(z_images)):
                if i == j:
                    continue

                z2, path2 = z_images[j]
                z_diff = z2 - z1

                if L_min <= abs(z_diff) <= L_max:
                    normalized_diff = z_diff / L_max
                    pair_key = (path1, path2)

                    if pair_key not in used_pairs:
                        valid_pairs.append((path1, path2, normalized_diff))
                        used_pairs.add(pair_key)

    return valid_pairs

def split_train_test_groups(
    all_group_names: List[str],
    group_to_patches: Dict[str, List[str]],
    exclude_groups: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    按 group 粒度划分训练/测试集。
    如果 exclude_groups 非空，则匹配到的 group 下所有 patches 强制进入测试集。

    :param all_group_names: 所有唯一的 group_name 列表
    :param group_to_patches: {group_name: [compound_id, ...]} 映射
    :param exclude_groups: 要排除的 group 名称列表（每个元素做子串匹配，匹配到的 group 不进训练集）
    :param train_ratio: 训练集占比
    :param random_seed: 随机种子
    :return: (train_compound_ids, test_compound_ids)
    """
    exclude_groups = exclude_groups or []

    # 先分出排除组
    matched = []
    for g in all_group_names:
        if any(exc in g for exc in exclude_groups):
            matched.append(g)
    remaining = [g for g in all_group_names if g not in matched]

    # 对剩余 group 按比例随机划分
    random.seed(random_seed)
    shuffled = random.sample(remaining, len(remaining))
    split_idx = int(len(shuffled) * train_ratio)
    train_group_names = shuffled[:split_idx]
    test_group_names = shuffled[split_idx:] + matched  # 排除组全进测试

    # 展开为 compound_id 列表
    def expand(names):
        ids = []
        for n in names:
            ids.extend(group_to_patches.get(n, []))
        return ids

    train_ids = expand(train_group_names)
    test_ids = expand(test_group_names)

    print(f"总 group 数: {len(all_group_names)}")
    print(f"  排除组: {matched if matched else '无'}")
    print(f"  训练组: {len(train_group_names)} 个 → {len(train_ids)} 个块")
    print(f"  测试组: {len(test_group_names)} 个 → {len(test_ids)} 个块")
    print(f"训练组示例: {sorted(train_group_names)[:5]}")
    print(f"测试组示例: {sorted(test_group_names)[:5]}")

    return train_ids, test_ids

def filter_pairs_by_max_limit(pairs: List[Tuple[str, str, float]], max_pairs: int, set_name: str, random_seed: int = 42) -> List[Tuple[str, str, float]]:
    """
    按上限过滤配对数量：超过上限则随机筛选，否则保留全部
    
    :param pairs: 原始配对列表
    :param max_pairs: 最大配对数量上限
    :param set_name: 数据集名称（train/test），用于打印日志
    :param random_seed: 随机种子（保证可重现）
    :return: 过滤后的配对列表
    """
    random.seed(random_seed)
    original_count = len(pairs)
    
    if original_count <= max_pairs:
        print(f"{set_name}集原始配对数({original_count}) ≤ 上限({max_pairs})，保留全部")
        return pairs
    else:
        filtered_pairs = random.sample(pairs, max_pairs)
        print(f"{set_name}集原始配对数({original_count}) > 上限({max_pairs})，随机筛选到{max_pairs}条")
        return filtered_pairs

def save_pairs_to_file(pairs: List[Tuple[str, str, float]], output_file: Union[str, Path]):
    """
    将配对信息保存到txt文件（制表符分隔）
    
    :param pairs: 配对列表
    :param output_file: 输出文件路径
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for img1, img2, diff in pairs:
            f.write(f"{img1}\t{img2}\t{diff:.6f}\n")
    
    print(f"\n已保存 {len(pairs)} 条配对到: {output_file}")

def main():
    """主函数：配置参数 + 执行流程"""
    # ===================== 可配置参数 =====================
    DATA_ROOT = "/data1/azt/cv/recoverZ/data_live_cell"
    INPUT_DIR = f"{DATA_ROOT}/images"          # 数据集根路径（所有group子文件夹所在目录）
    OUTPUT_DIR = f"{DATA_ROOT}/pairs"          # 配对文件输出目录
    L_MIN = 1                                  # 最小z差值（绝对值）
    L_MAX = 20                                 # 最大z差值（绝对值）
    TRAIN_RATIO = 0.9                          # 训练集占比
    RANDOM_SEED = 4                           # 随机种子（保证结果可重现）
    TRAIN_MAX_PAIRS = 150000                   # 训练集配对数量上限
    TEST_MAX_PAIRS = 5000                      # 测试集配对数量上限
    EXCLUDE_GROUPS = ["plane 6 z-stack"]  # 要从训练集中排除的group名称列表（子串匹配，设为空列表则不排除）
    # ======================================================

    print("===== 开始收集图像组信息 =====")
    try:
        image_groups, all_group_names, group_to_patches = collect_image_groups(INPUT_DIR)
    except ValueError as e:
        print(e)
        return

    train_ids, test_ids = split_train_test_groups(
        all_group_names,
        group_to_patches,
        exclude_groups=EXCLUDE_GROUPS,
        train_ratio=TRAIN_RATIO,
        random_seed=RANDOM_SEED
    )

    print("\n===== 生成训练集配对 =====")
    train_pairs_original = generate_all_possible_pairs(
        image_groups,
        target_compound_ids=train_ids,
        L_min=L_MIN,
        L_max=L_MAX
    )
    train_pairs = filter_pairs_by_max_limit(
        train_pairs_original,
        max_pairs=TRAIN_MAX_PAIRS,
        set_name="训练",
        random_seed=RANDOM_SEED
    )

    print("\n===== 生成测试集配对 =====")
    test_pairs_original = generate_all_possible_pairs(
        image_groups,
        target_compound_ids=test_ids,
        L_min=L_MIN,
        L_max=L_MAX
    )
    test_pairs = filter_pairs_by_max_limit(
        test_pairs_original,
        max_pairs=TEST_MAX_PAIRS,
        set_name="测试",
        random_seed=RANDOM_SEED
    )

    train_output_file = os.path.join(OUTPUT_DIR, "train_pairs.txt")
    test_output_file = os.path.join(OUTPUT_DIR, "val_pairs.txt")

    save_pairs_to_file(train_pairs, train_output_file)
    save_pairs_to_file(test_pairs, test_output_file)

    print("\n===== 最终汇总 =====")
    print(f"训练集配对文件: {train_output_file} ({len(train_pairs)} 行)")
    print(f"测试集配对文件: {test_output_file} ({len(test_pairs)} 行)")
    print(f"总配对数: {len(train_pairs) + len(test_pairs)}")
    print(f"配置参数：L_MIN={L_MIN}, L_MAX={L_MAX}, 训练上限={TRAIN_MAX_PAIRS}, 测试上限={TEST_MAX_PAIRS}")
    print(f"排除组: {'无' if not EXCLUDE_GROUPS else EXCLUDE_GROUPS}")

    if train_pairs:
        print("\n===== 训练集配对示例 =====")
        for i, (img1, img2, diff) in enumerate(train_pairs[:3]):
            # 从路径中提取 group 和 patch 信息
            parts = Path(img1).relative_to(INPUT_DIR).parts
            ctx = "/".join(parts[:-1])  # group/patch_id
            print(f"示例{i+1}: {ctx}/{Path(img1).name} → {Path(img2).name} (diff: {diff:.6f})")

    if test_pairs:
        print("\n===== 测试集配对示例 =====")
        for i, (img1, img2, diff) in enumerate(test_pairs[:3]):
            parts = Path(img1).relative_to(INPUT_DIR).parts
            ctx = "/".join(parts[:-1])
            print(f"示例{i+1}: {ctx}/{Path(img1).name} → {Path(img2).name} (diff: {diff:.6f})")

if __name__ == "__main__":
    main()
