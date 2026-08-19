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

def process_single_tiff_to_png(tiff_path, output_root,
                                filter_type="gaussian", kernel_size=3, sigmaX=1.0,
                                n_rows=2, n_cols=2, slice_step=1):
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
        slice_step: 每隔 slice_step 层保存一层
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
        for slice_idx in tqdm(range(0, num_slices, slice_step), desc=f"处理 {tif_basename}", unit="层"):
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
    # ===================== 可配置参数 =====================
    DATA_ROOT = "data_live_cell_0625"
    TIFF_PATH = f"{DATA_ROOT}/tif/2026-0625-for_Zitian/7/7-Z_stack-channel_2.tif"  # 单个TIF文件路径
    OUTPUT_ROOT = f"{DATA_ROOT}/images"
    FILTER_TYPE = "gaussian"
    KERNEL_SIZE = 5
    SIGMA_X = 1.0
    N_ROWS = 4          # 行方向切块数
    N_COLS = 4          # 列方向切块数
    SLICE_STEP = 1      # 每隔 SLICE_STEP 层保存一层
    # ======================================================

    if not os.path.exists(TIFF_PATH):
        print(f"❌ TIF文件不存在：{TIFF_PATH}")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"📌 输入文件：{TIFF_PATH}")
    print(f"📌 滤波配置：类型={FILTER_TYPE}，核大小={KERNEL_SIZE}，高斯标准差={SIGMA_X}")
    print(f"📌 保留策略：每隔 {SLICE_STEP} 层保存一层，输出文件名连续（z1, z2, z3, ...）")
    print(f"📌 图像分块：每个图像分成 {N_ROWS}x{N_COLS} = {N_ROWS * N_COLS} 个块")

    success, total_slices = process_single_tiff_to_png(
        TIFF_PATH, OUTPUT_ROOT,
        filter_type=FILTER_TYPE,
        kernel_size=KERNEL_SIZE,
        sigmaX=SIGMA_X,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        slice_step=SLICE_STEP
    )

    print("\n" + "=" * 60)
    print("📊 处理完成统计结果")
    print(f"   输入文件：{TIFF_PATH}")
    print(f"   {'✅ 处理成功' if success else '❌ 处理失败'}")
    print(f"   🖼️  总输出PNG切片数：{total_slices}")
    print(f"   🧹 去噪配置：{FILTER_TYPE}滤波（核大小={KERNEL_SIZE}）")
    print(f"   💾 所有PNG文件保存至：{OUTPUT_ROOT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
