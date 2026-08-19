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
    对TIFF数据进行全局归一化处理：所有时间帧和通道共用同一个最大值
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

def process_timeseries_tiff_to_png(tiff_path, output_root,
                                    filter_type="gaussian", kernel_size=3, sigmaX=1.0,
                                    n_rows=2, n_cols=2, time_step=1):
    """
    处理时间序列TIFF文件：读取→归一化→去噪→分块→按时间戳保存为PNG

    自动适配两种维度格式：
      - (T, H, W):  单通道时间序列（如已预先拆分通道的文件，本任务即此类型）
      - (T, C, H, W): 多通道时间序列，每个通道独立保存

    目录结构（单通道）：
      output_root / tif文件名 / patch_{idx} / t{t}.png
    目录结构（多通道）：
      output_root / tif文件名 / patch_{idx} / t{t}_c{c}.png

    参数：
        tiff_path: 输入TIFF文件的完整路径
        output_root: 输出根目录
        filter_type: 滤波类型（gaussian/median/none）
        kernel_size: 滤波核大小（奇数）
        sigmaX: 高斯滤波标准差
        n_rows, n_cols: 行列方向切块数
        time_step: 每隔 time_step 个时间帧保存一帧
    返回：(是否成功, 输出的PNG切片总数)
    """

    try:
        tiff_data = tiff.imread(tiff_path)
        print(f"原始TIFF形状: {tiff_data.shape}, dtype: {tiff_data.dtype}")

        # 统一为 (T, C, H, W) 格式处理
        if tiff_data.ndim == 2:
            # 单张图 (H, W) → (1, 1, H, W)
            tiff_data = tiff_data[np.newaxis, np.newaxis, ...]
        elif tiff_data.ndim == 3:
            # (T, H, W) → (T, 1, H, W)，视为单通道时间序列
            tiff_data = tiff_data[:, np.newaxis, ...]
        elif tiff_data.ndim == 4:
            # (T, C, H, W) 直接使用
            pass
        else:
            raise ValueError(f"不支持的TIFF维度数: {tiff_data.ndim} (shape={tiff_data.shape})")

        num_frames, num_channels, H, W = tiff_data.shape
        print(f"解析维度 → 时间帧数 T={num_frames}, 通道数 C={num_channels}, H={H}, W={W}")

        # 全局归一化（所有帧 + 所有通道共用同一最大值，保留时间方向上的相对强度）
        normalized_data = normalize_tiff_data(tiff_data)

        # 以 tif 文件名（不含扩展名）创建输出文件夹
        tif_basename = os.path.splitext(os.path.basename(tiff_path))[0]
        tif_out_dir = os.path.join(output_root, tif_basename)
        os.makedirs(tif_out_dir, exist_ok=True)

        total_output_slices = 0
        multi_channel = num_channels > 1

        # 每隔 time_step 帧保存一帧，输出文件名连续（t1, t2, t3, ...）
        output_idx = 1
        for frame_idx in tqdm(range(0, num_frames, time_step),
                              desc=f"处理 {tif_basename}", unit="帧"):
            for c_idx in range(num_channels):
                slice_data = normalized_data[frame_idx, c_idx]
                denoised_slice = denoise_slice(slice_data, filter_type, kernel_size, sigmaX)

                # 将当前帧分成 n_rows x n_cols 个块
                patches = split_image_into_patches(denoised_slice, n_rows, n_cols)

                # 为每个块保存到 tif_name/patch_{idx}/t{t}.png
                for patch_idx, patch in enumerate(patches):
                    patch_dir = os.path.join(tif_out_dir, f"patch_{patch_idx}")
                    os.makedirs(patch_dir, exist_ok=True)

                    if multi_channel:
                        slice_name = f"t{output_idx}_c{c_idx + 1}.png"
                    else:
                        slice_name = f"t{output_idx}.png"
                    slice_path = os.path.join(patch_dir, slice_name)

                    cv2.imwrite(slice_path, patch)

            output_idx += 1
            total_output_slices += n_rows * n_cols * num_channels

        return True, total_output_slices

    except Exception as e:
        print(f"\n❌ 处理文件失败 {tiff_path}：{str(e)}")
        return False, 0

def main():
    # ===================== 可配置参数 =====================
    DATA_ROOT = "data_live_cell_0625"
    TIFF_PATH = f"{DATA_ROOT}/tif/2026-0625-for_Zitian/7/7-time-channel_2.tif"  # 时间序列TIF文件
    OUTPUT_ROOT = f"{DATA_ROOT}/images_ts"
    FILTER_TYPE = "gaussian"
    KERNEL_SIZE = 5
    SIGMA_X = 1.0
    N_ROWS = 4          # 行方向切块数
    N_COLS = 4          # 列方向切块数
    TIME_STEP = 1       # 每隔 TIME_STEP 帧保存一帧
    # ======================================================

    if not os.path.exists(TIFF_PATH):
        print(f"❌ TIF文件不存在：{TIFF_PATH}")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"📌 输入文件：{TIFF_PATH}")
    print(f"📌 滤波配置：类型={FILTER_TYPE}，核大小={KERNEL_SIZE}，高斯标准差={SIGMA_X}")
    print(f"📌 时间采样：每隔 {TIME_STEP} 帧保存一帧，输出文件名连续（t1, t2, t3, ...）")
    print(f"📌 图像分块：每帧分成 {N_ROWS}x{N_COLS} = {N_ROWS * N_COLS} 个块")

    success, total_slices = process_timeseries_tiff_to_png(
        TIFF_PATH, OUTPUT_ROOT,
        filter_type=FILTER_TYPE,
        kernel_size=KERNEL_SIZE,
        sigmaX=SIGMA_X,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        time_step=TIME_STEP
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
