import os
import numpy as np
import tifffile as tiff
import warnings
from tqdm import tqdm  # 导入进度条库

warnings.filterwarnings('ignore')


def extract_core_identifier(tiff_filename):
    """
    从TIFF文件名提取核心标识及拆分信息
    返回值：(完整标识, 前缀部分, s后缀部分)
    示例：a01_s1 → (a01_s1, a01, s1)；b02_s2 → (b02_s2, b02, s2)
    适配规则：任意单个英文字母+数字 + 紧邻的s+数字组合
    """
    # 去掉文件后缀
    name_without_ext = os.path.splitext(tiff_filename)[0]
    # 按下划线分割
    parts = name_without_ext.split('_')
    # 寻找「单个字母+数字」 + 紧邻的「s+数字」组合
    core_parts = []
    s_suffix = ""
    prefix_part = ""
    for i, part in enumerate(parts):
        # 匹配：单个字母+数字（如a01、b02、c03）
        if len(part) >= 2 and part[0].isalpha() and part[1:].isdigit():
            prefix_part = part  # 提取前缀（如a01、b02）
            core_parts.append(part)
            # 检查下一个部分是否是s+数字（如s1、s2）
            if i+1 < len(parts) and parts[i+1].startswith('s') and parts[i+1][1:].isdigit():
                s_suffix = parts[i+1]  # 提取s后缀（如s1、s2）
                core_parts.append(s_suffix)
                break
    if len(core_parts) != 2:
        raise ValueError(f"无法从文件名{tiff_filename}提取「字母+数字_s+数字」格式的核心标识（如a01_s1、b02_s2）")
    full_identifier = '_'.join(core_parts)
    return full_identifier, prefix_part, s_suffix


def check_w_suffix(tiff_filename, wanted_w_suffix='w1'):
    """
    检查文件名中aXX_sXX后的w后缀：保留指定的w后缀文件，跳过其他w后缀文件
    参数：
        wanted_w_suffix: 要保留的w后缀（'w1'或'w2'）
    返回：True（保留/匹配指定w后缀）、False（跳过/不匹配）
    """
    # 校验输入的w后缀合法性
    if wanted_w_suffix not in ['w1', 'w2']:
        raise ValueError(f"wanted_w_suffix只能是'w1'或'w2'，当前输入：{wanted_w_suffix}")
    
    name_without_ext = os.path.splitext(tiff_filename)[0]
    parts = name_without_ext.split('_')
    
    # 找到aXX_sXX的位置，检查其后的w后缀
    for i, part in enumerate(parts):
        # 找到sXX的位置，检查下一个部分是否是w开头
        if part.startswith('s') and part[1:].isdigit() and i+1 < len(parts):
            next_part = parts[i+1]
            if next_part.startswith(wanted_w_suffix):
                return True  # 匹配指定w后缀：保留处理
            elif next_part.startswith('w1') or next_part.startswith('w2'):
                return False  # 其他w后缀：跳过
    # 无明确w1/w2后缀：默认跳过（可选调整为True）
    return False


def natural_sort_key(s):
    """自然排序函数（1,2,10而非1,10,2）"""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def collect_tiff_files_by_identifier(input_root, wanted_w_suffix='w1'):
    """
    递归收集所有TIFF文件，按核心标识分组（仅处理指定w后缀的文件，跳过其他w后缀）
    参数：
        wanted_w_suffix: 要保留的w后缀（'w1'或'w2'）
    返回：identifier_info = {full_identifier: (prefix_part, s_suffix, z_layer_files)}
    """
    # 1. 获取并排序z层子文件夹（按z_00、z_01、z_02...顺序）
    subfolders = []
    for item in os.listdir(input_root):
        item_path = os.path.join(input_root, item)
        if os.path.isdir(item_path) and item.startswith('BBBC006_v1_images_z_'):
            subfolders.append(item_path)
    # 按z后的数字自然排序
    subfolders.sort(key=lambda x: natural_sort_key(os.path.basename(x).split('_z_')[-1]))
    if not subfolders:
        raise ValueError(f"在{input_root}下未找到BBBC006_v1_images_z_XX格式的子文件夹！")
    print(f"📌 找到{len(subfolders)}个z层文件夹：{[os.path.basename(f) for f in subfolders]}")

    # 2. 按核心标识分组（仅处理指定w后缀的文件，跳过其他w后缀）
    identifier_info = {}
    unwanted_w_skip_count = 0  # 统计跳过的非目标w后缀文件数（通用命名，不再固定w2）
    unwanted_w_type = 'w2' if wanted_w_suffix == 'w1' else 'w1'  # 确定要跳过的w后缀类型
    
    for z_idx, folder_path in enumerate(subfolders):
        # 遍历当前z层文件夹下的所有TIFF文件
        tiff_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith((".tif", ".tiff"))]
        if not tiff_files:
            print(f"⚠️ {os.path.basename(folder_path)}下未找到TIFF文件，跳过")
            continue
        
        for tiff_path in tiff_files:
            tiff_filename = os.path.basename(tiff_path)
            
            # 第一步：过滤非目标w后缀的文件
            if not check_w_suffix(tiff_filename, wanted_w_suffix):
                unwanted_w_skip_count += 1
                print(f"⚠️ 跳过_{unwanted_w_type}文件：{tiff_filename}")
                continue
            
            # 第二步：提取核心标识（仅处理目标w后缀的文件）
            try:
                full_id, prefix_part, s_suffix = extract_core_identifier(tiff_filename)
            except ValueError as e:
                print(f"⚠️ 跳过文件{tiff_filename}：{e}")
                continue
            
            # 初始化分组信息
            if full_id not in identifier_info:
                identifier_info[full_id] = {
                    "prefix": prefix_part,
                    "s_suffix": s_suffix,
                    "z_files": [None] * len(subfolders)
                }
            # 填充对应z层的文件路径
            identifier_info[full_id]["z_files"][z_idx] = tiff_path

    print(f"\n📌 本次共跳过 {unwanted_w_skip_count} 个_{unwanted_w_type}格式的文件")
    return identifier_info, subfolders


def load_z_stack(identifier, z_layer_files):
    """
    加载单个核心标识的所有z层数据，合并为3D stack (z, h, w)
    【修改】移除归一化到0-255和强制转换uint8的逻辑，保留原始数据类型和数值范围
    """
    z_stack_data = []
    valid_z_layers = []  # 记录有效z层索引
    for z_idx, tiff_path in enumerate(z_layer_files):
        if tiff_path is None:
            print(f"⚠️ {identifier} z_{z_idx:02d}层无对应文件，跳过")
            continue
        try:
            # 读取TIFF文件（支持单页/多页，但仅取第一页）
            tiff_data = tiff.imread(tiff_path)
            # 数据格式处理：仅确保是2D单通道（不修改数值和数据类型）
            if len(tiff_data.shape) == 3:
                if tiff_data.shape[0] == 1:
                    tiff_data = tiff_data[0]  # 去掉多余维度
                else:
                    raise ValueError(f"不支持多通道数据：{tiff_data.shape}")
            if len(tiff_data.shape) != 2:
                raise ValueError(f"图像维度异常（需2D）：{tiff_data.shape}")
            
            # 【已删除】归一化到0-255 + 强制转换uint8的逻辑
            # 保留原始数据类型和数值范围，直接添加到z-stack
            z_stack_data.append(tiff_data)
            valid_z_layers.append(z_idx)
        except Exception as e:
            print(f"⚠️ {identifier} z_{z_idx:02d}层读取失败：{str(e)}")
            continue
    
    if not z_stack_data:
        return None
    # 转换为3D numpy数组 (z, height, width)，保留原始数据类型
    return np.stack(z_stack_data, axis=0), valid_z_layers


def save_merged_tiff(prefix_part, s_suffix, z_stack_data, output_root):
    """
    保存合并后的3D z-stack为单个TIFF文件（不再创建子文件夹）
    - 保存路径：output_root/s_suffix/prefix_part.tif（如output_root/s1/a01.tif）
    - 【修改】保留原始数据类型保存，不强制uint8
    """
    # 1. 构建保存目录（仅创建s_suffix目录，如s1、s2）
    output_dir = os.path.join(output_root, s_suffix)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 构建TIFF文件名（如a01.tif）
    tiff_filename = f"{prefix_part}.tif"
    tiff_path = os.path.join(output_dir, tiff_filename)
    
    try:
        # 保存3D z-stack为单个TIFF文件，保留原始数据类型
        tiff.imwrite(tiff_path, z_stack_data)
        return True
    except Exception as e:
        print(f"⚠️ {tiff_filename} 保存失败：{str(e)}")
        return False


def main():
    # ---------------------- 配置参数（按需修改） ----------------------
    INPUT_ROOT = "data_BBBC006/tif_raw"
    OUTPUT_ROOT = "data_BBBC006/tif"  # TIFF输出根目录
    W_SUFFIX_TO_KEEP = "w2"  # 可选值：'w1' 或 'w2'，指定要保留的w后缀文件
    # ------------------------------------------------------------------

    try:
        # 校验W_SUFFIX_TO_KEEP的合法性
        if W_SUFFIX_TO_KEEP not in ['w1', 'w2']:
            raise ValueError(f"配置参数W_SUFFIX_TO_KEEP只能是'w1'或'w2'，当前值：{W_SUFFIX_TO_KEEP}")
        
        # 1. 收集并分组TIFF文件（传入要保留的w后缀参数）
        print(f"📌 开始收集并分组TIFF文件（仅处理_{W_SUFFIX_TO_KEEP}格式，跳过其他w后缀）...")
        identifier_info, z_folders = collect_tiff_files_by_identifier(INPUT_ROOT, wanted_w_suffix=W_SUFFIX_TO_KEEP)
        total_identifiers = len(identifier_info)
        if total_identifiers == 0:
            print(f"❌ 未找到可处理的TIFF文件（仅_{W_SUFFIX_TO_KEEP}格式），程序退出")
            return
        
        print(f"📌 共找到 {total_identifiers} 个可处理的标识（_{W_SUFFIX_TO_KEEP}格式，如a01_s1）\n")

        # 2. 批量处理每个标识的z-stack（带进度条）
        success_count = 0
        fail_count = 0
        fail_identifiers = []

        with tqdm(total=total_identifiers, desc="处理进度", unit="标识") as pbar:
            for full_id, info in identifier_info.items():
                prefix_part = info["prefix"]
                s_suffix = info["s_suffix"]
                z_layer_files = info["z_files"]
                
                # 加载当前标识的所有z层数据
                z_stack_result = load_z_stack(full_id, z_layer_files)
                if z_stack_result is None:
                    fail_count += 1
                    fail_identifiers.append(full_id)
                    pbar.update(1)
                    continue
                z_stack_data, valid_layers = z_stack_result
                
                # 保存为单个TIFF文件（保留原始数据类型）
                save_success = save_merged_tiff(prefix_part, s_suffix, z_stack_data, OUTPUT_ROOT)
                if save_success:
                    success_count += 1
                else:
                    fail_count += 1
                    fail_identifiers.append(full_id)
                pbar.update(1)

        # 3. 输出最终统计结果
        print("\n" + "="*50)
        print("📊 处理完成统计：")
        print(f"   目标保留后缀：_{W_SUFFIX_TO_KEEP}")
        print(f"   成功生成：{success_count} 个TIFF文件")
        print(f"   处理失败：{fail_count} 个标识")
        if fail_count > 0:
            print(f"   失败标识：{fail_identifiers}")
        print(f"   输出根目录：{OUTPUT_ROOT}（按s1/s2子目录拆分，每个标识对应一个TIFF文件）")
        print("="*50)

    except Exception as e:
        print(f"\n❌ 程序执行异常：{str(e)}")
        raise  # 抛出异常便于调试


if __name__ == "__main__":
    main()