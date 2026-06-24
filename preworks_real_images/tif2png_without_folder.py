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

def split_image_into_patches(image, n_rows, n_cols):
    """
    将图像分成 n_rows x n_cols 个块。
    要求 image 的 H 能被 n_rows 整除，W 能被 n_cols 整除。
    返回按行优先排列的块列表。
    """
    H, W = image.shape
    if H % n_rows != 0 or W % n_cols != 0:
        raise ValueError(f"图像尺寸 ({H}x{W}) 无法被 {n_rows}x{n_cols} 整除")
    patch_h = H // n_rows
    patch_w = W // n_cols
    patches = []
    for row in range(n_rows):
        for col in range(n_cols):
            patch = image[row * patch_h:(row + 1) * patch_h,
                          col * patch_w:(col + 1) * patch_w]
            patches.append(patch)
    return patches

def process_tiff_to_png(tiff_path, output_root,
                         filter_type="gaussian", kernel_size=3, sigmaX=1.0,
                         n_rows=2, n_cols=2, slice_step=2):
    """
    处理单个TIFF文件：读取→归一化→去噪→分块→逐切片保存为PNG

    目录结构：output_root / tif文件名 / patch_{idx} / z{idx}.png

    参数：
        tiff_path: 输入TIFF文件的完整路径
        output_root: 输出根目录
        filter_type: 滤波类型（gaussian/median/none）
        kernel_size: 滤波核大小（奇数）
        sigmaX: 高斯滤波标准差
        n_rows, n_cols: 行列方向切块数
        slice_step: 每隔 slice_step 层保存一层（默认2，即奇数层）
    返回：(是否成功, 输出的PNG切片总数)
    """

    try:
        tiff_data = tiff.imread(tiff_path)

        if len(tiff_data.shape) == 2:
            tiff_data = np.expand_dims(tiff_data, axis=0)

        normalized_data = normalize_tiff_data(tiff_data)

        num_slices = normalized_data.shape[0]

        # 以 tif 文件名（不含扩展名）创建输出文件夹
        tif_basename = os.path.splitext(os.path.basename(tiff_path))[0]
        tif_out_dir = os.path.join(output_root, tif_basename)
        os.makedirs(tif_out_dir, exist_ok=True)
        
        total_output_slices = 0
        
        # 每隔 slice_step 层保存一层，输出文件名连续（z1, z2, z3, ...）
        output_idx = 1
        for slice_idx in range(0, num_slices, slice_step):
            slice_data = normalized_data[slice_idx]
            denoised_slice = denoise_slice(slice_data, filter_type, kernel_size, sigmaX)

            # 将切片分成 n_rows x n_cols 个块
            patches = split_image_into_patches(denoised_slice, n_rows, n_cols)
            
            # 为每个块保存到 tif_name/patch_{idx}/z{idx}.png
            for patch_idx, patch in enumerate(patches):
                patch_dir = os.path.join(tif_out_dir, f"patch_{patch_idx}")
                os.makedirs(patch_dir, exist_ok=True)

                slice_name = f"z{output_idx}.png"
                slice_path = os.path.join(patch_dir, slice_name)

                cv2.imwrite(slice_path, patch)

            output_idx += 1
            total_output_slices += n_rows * n_cols

        return True, total_output_slices
    
    except Exception as e:
        print(f"\n❌ 处理文件失败 {tiff_path}：{str(e)}")
        return False, 0

def main():
    DATA_ROOT = "data_live_cell"
    INPUT_ROOT = f"{DATA_ROOT}/tif/2026-0609/2026.06.11 live cell-hela/z-stack"
    OUTPUT_ROOT = f"{DATA_ROOT}/images"
    FILTER_TYPE = "gaussian"
    KERNEL_SIZE = 5
    SIGMA_X = 1.0
    N_ROWS = 4          # 行方向切块数
    N_COLS = 4          # 列方向切块数
    SLICE_STEP = 1      # 每隔 SLICE_STEP 层保存一层

    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 输入目录不存在：{INPUT_ROOT}")
        return
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 获取所有tif文件，按文件名中的数字前缀排序
    all_tif_files = []
    for item in os.listdir(INPUT_ROOT):
        item_path = os.path.join(INPUT_ROOT, item)
        if os.path.isfile(item_path) and item.lower().endswith((".tif", ".tiff")):
            # 提取文件名开头的数字（如 1_488_Em525_Widefield_.tif -> 1）
            number_str = item.split('_')[0]
            try:
                file_num = int(number_str)
                all_tif_files.append((file_num, item_path))
            except ValueError:
                all_tif_files.append((float('inf'), item_path))
    
    # 按数字排序
    all_tif_files.sort(key=lambda x: x[0])
    
    if not all_tif_files:
        print(f"⚠️  {INPUT_ROOT} 中未找到tif文件")
        return
    
    total_files = 0
    success_files = 0
    total_slices = 0
    
    print(f"📌 滤波配置：类型={FILTER_TYPE}，核大小={KERNEL_SIZE}，高斯标准差={SIGMA_X}")
    print(f"📌 保留策略：每隔 {SLICE_STEP} 层保存一层，输出文件名连续（z1, z2, z3, ...）")
    print(f"📌 图像分块：每个图像分成 {N_ROWS}x{N_COLS} = {N_ROWS * N_COLS} 个块")
    print(f"📌 共找到 {len(all_tif_files)} 个tif文件需要处理")
    
    for file_num, tiff_file in tqdm(all_tif_files, desc="处理tif文件"):
        total_files += 1
        success, slices = process_tiff_to_png(
            tiff_file, OUTPUT_ROOT,
            filter_type=FILTER_TYPE,
            kernel_size=KERNEL_SIZE,
            sigmaX=SIGMA_X,
            n_rows=N_ROWS,
            n_cols=N_COLS,
            slice_step=SLICE_STEP
        )
        if success:
            success_files += 1
            total_slices += slices
            
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
