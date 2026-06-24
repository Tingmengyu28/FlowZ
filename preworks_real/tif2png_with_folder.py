import os
import numpy as np
import tifffile as tiff
import cv2
import warnings
from tqdm import tqdm

# 忽略无关警告
warnings.filterwarnings('ignore')

def normalize_tiff_data(tiff_data):
    """
    对TIFF数据进行归一化处理：每个像素值除以整个TIFF的最大值
    处理边界情况（如全0数据），避免除以0错误
    """
    max_val = tiff_data.max()
    if max_val == 0:
        normalized_data = np.zeros_like(tiff_data, dtype=np.float32)
    else:
        normalized_data = tiff_data.astype(np.float32) / max_val
    normalized_data = (normalized_data * 255).astype(np.uint8)
    return normalized_data

def denoise_slice(slice_data, filter_type="gaussian", kernel_size=3, sigmaX=1.0):
    """
    对单张切片进行去噪滤波处理（归一化后调用）
    参数说明：
        slice_data: 单切片的uint8格式数据（h, w）
        filter_type: 滤波类型，可选 "gaussian"（高斯滤波，适合高斯噪声）、"median"（中值滤波，适合椒盐噪声）
        kernel_size: 滤波核大小（必须是奇数，如3/5/7），核越大去噪越强但可能模糊细节
        sigmaX: 高斯滤波的X方向标准差（仅高斯滤波生效），越大平滑效果越强
    返回：去噪后的uint8格式切片数据
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
        print(f"⚠️  滤波核大小需为奇数，已自动调整为：{kernel_size}")
    
    try:
        if filter_type == "gaussian":
            denoised_data = cv2.GaussianBlur(slice_data, (kernel_size, kernel_size), sigmaX)
        elif filter_type == "median":
            denoised_data = cv2.medianBlur(slice_data, kernel_size)
        else:
            print(f"⚠️  未知滤波类型 {filter_type}，使用原始数据")
            denoised_data = slice_data
        return denoised_data
    except Exception as e:
        print(f"\n❌ 滤波处理失败：{str(e)}，使用原始数据")
        return slice_data

def split_image_into_4_patches(image):
    """
    将2048x2048的图像分成4个512x512的块
    布局：
    +-------+-------+
    |   0   |   1   |
    | 512x512 | 512x512 |
    +-------+-------+
    |   2   |   3   |
    | 512x512 | 512x512 |
    +-------+-------+
    """
    patches = []
    patch_size = 512
    for i in range(4):
        row = i // 2
        col = i % 2
        start_row = row * patch_size
        end_row = start_row + patch_size
        start_col = col * patch_size
        end_col = start_col + patch_size
        patch = image[start_row:end_row, start_col:end_col]
        patches.append(patch)
    return patches

def process_tiff_to_png(tiff_path, output_root, start_folder_idx, filter_type="gaussian", kernel_size=3, sigmaX=1.0):
    """
    处理单个TIFF文件：读取→归一化→去噪→分成4块→逐切片保存为PNG
    参数：
        tiff_path: 输入TIFF文件的完整路径
        output_root: 输出根目录
        start_folder_idx: 起始文件夹索引（每个tif对应4个文件夹）
        filter_type: 滤波类型（gaussian/median/none）
        kernel_size: 滤波核大小（奇数）
        sigmaX: 高斯滤波标准差
    返回：(是否成功, 输出的PNG切片总数)
    """
    
    try:
        tiff_data = tiff.imread(tiff_path)
        
        if len(tiff_data.shape) == 2:
            tiff_data = np.expand_dims(tiff_data, axis=0)
        
        normalized_data = normalize_tiff_data(tiff_data)
        
        num_slices = normalized_data.shape[0]
        
        total_output_slices = 0
        
        # 保留原始TIFF的奇数层（1, 3, 5, ..., 81），但输出文件名是连续的 z1, z2, z3, ...
        output_idx = 1
        for slice_idx in range(0, num_slices, 2):
            slice_data = normalized_data[slice_idx]
            denoised_slice = denoise_slice(slice_data, filter_type, kernel_size, sigmaX)
            
            # 将512x512的切片分成4个块
            patches = split_image_into_4_patches(denoised_slice)
            
            # 为每个块保存
            for patch_idx, patch in enumerate(patches):
                folder_idx = start_folder_idx + patch_idx
                output_dir = os.path.join(output_root, str(folder_idx))
                os.makedirs(output_dir, exist_ok=True)
                
                slice_name = f"z{output_idx}.png"
                slice_path = os.path.join(output_dir, slice_name)
                
                cv2.imwrite(slice_path, patch)
            
            output_idx += 1
            total_output_slices += 4
        
        return True, total_output_slices
    
    except Exception as e:
        print(f"\n❌ 处理文件失败 {tiff_path}：{str(e)}")
        return False, 0

def main():
    DATA_ROOT = "data_brain"
    INPUT_ROOT = f"{DATA_ROOT}/tif/0505"
    OUTPUT_ROOT = f"{DATA_ROOT}/images"
    FILTER_TYPE = "gaussian"
    KERNEL_SIZE = 1
    SIGMA_X = 1.0

    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 输入目录不存在：{INPUT_ROOT}")
        return
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 获取所有tif文件夹，按数字顺序排序
    all_tif_folders = []
    for item in os.listdir(INPUT_ROOT):
        item_path = os.path.join(INPUT_ROOT, item)
        if os.path.isdir(item_path):
            # 提取数字
            for prefix in ['insituVc', 'InsituVc', 'insituvc']:
                if item.startswith(prefix):
                    number_str = item[len(prefix):]
                    try:
                        folder_num = int(number_str)
                        all_tif_folders.append((folder_num, item))
                        break
                    except ValueError:
                        pass
    
    # 按数字排序
    all_tif_folders.sort(key=lambda x: x[0])
    
    if not all_tif_folders:
        print(f"⚠️  {INPUT_ROOT} 中未找到tif文件夹")
        return
    
    total_files = 0
    success_files = 0
    total_slices = 0
    
    print(f"📌 滤波配置：类型={FILTER_TYPE}，核大小={KERNEL_SIZE}，高斯标准差={SIGMA_X}")
    print("📌 保留策略：原始TIFF的奇数层（1, 3, 5, ..., 81），输出文件名连续（z1, z2, z3, ...）")
    print("📌 图像分块：每个2048x2048图像分成4个512x512块")
    print(f"📌 共找到 {len(all_tif_folders)} 个tif文件夹需要处理")
    
    current_folder_idx = 1
    for folder_num, folder_name in tqdm(all_tif_folders, desc="处理tif文件夹"):
        subfolder_path = os.path.join(INPUT_ROOT, folder_name)
        
        tiff_files = [
            os.path.join(subfolder_path, f) 
            for f in os.listdir(subfolder_path)
            if f.lower().endswith((".tif", ".tiff"))
        ]
        
        if not tiff_files:
            print(f"⚠️  文件夹 {folder_name} 中未找到TIFF文件")
            continue
        
        for tiff_file in tiff_files:
            total_files += 1
            success, slices = process_tiff_to_png(
                tiff_file, OUTPUT_ROOT, current_folder_idx,
                filter_type=FILTER_TYPE,
                kernel_size=KERNEL_SIZE,
                sigmaX=SIGMA_X
            )
            if success:
                success_files += 1
                total_slices += slices
            current_folder_idx += 4  # 每个tif对应4个文件夹
            
    print("\n" + "="*60)
    print("📊 处理完成统计结果")
    print(f"   总TIFF文件数：{total_files}")
    print(f"   ✅ 成功处理文件数：{success_files}")
    print(f"   🖼️  总输出PNG切片数：{total_slices}")
    print(f"   🧹 去噪配置：{FILTER_TYPE}滤波（核大小={KERNEL_SIZE}）")
    print(f"   💾 所有PNG文件保存至：{OUTPUT_ROOT}")
    print("="*60)

if __name__ == "__main__":
    main()
