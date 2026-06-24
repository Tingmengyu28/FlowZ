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
    # 获取数据最大值（防止全0时除以0）
    max_val = tiff_data.max()
    if max_val == 0:
        # 全0数据直接返回0
        normalized_data = np.zeros_like(tiff_data, dtype=np.float32)
    else:
        # 归一化到0-1范围
        normalized_data = tiff_data.astype(np.float32) / max_val
    # 转换为0-255的uint8格式（适配PNG保存和滤波处理）
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
    # 校验核大小（必须为奇数）
    if kernel_size % 2 == 0:
        kernel_size += 1  # 自动转为奇数
        print(f"⚠️  滤波核大小需为奇数，已自动调整为：{kernel_size}")
    
    try:
        if filter_type == "gaussian":
            # 高斯滤波（推荐：适合大部分平滑去噪场景）
            denoised_data = cv2.GaussianBlur(slice_data, (kernel_size, kernel_size), sigmaX)
        elif filter_type == "median":
            # 中值滤波（适合椒盐噪声/脉冲噪声）
            denoised_data = cv2.medianBlur(slice_data, kernel_size)
        else:
            # 无滤波（保留原始数据）
            print(f"⚠️  未知滤波类型 {filter_type}，使用原始数据")
            denoised_data = slice_data
        return denoised_data
    except Exception as e:
        print(f"\n❌ 滤波处理失败：{str(e)}，使用原始数据")
        return slice_data

def process_tiff_to_png(tiff_path, output_root, subfolder, filter_type="gaussian", kernel_size=3, sigmaX=1.0):
    """
    处理单个TIFF文件：读取→归一化→去噪→逐切片保存为PNG
    参数：
        tiff_path: 输入TIFF文件的完整路径
        output_root: 输出根目录
        subfolder: 子文件夹名（s1/s2）
        filter_type: 滤波类型（gaussian/median/none）
        kernel_size: 滤波核大小（奇数）
        sigmaX: 高斯滤波标准差
    """
    # 提取TIFF文件名（不含后缀），如a01.tif → a01
    tiff_filename = os.path.basename(tiff_path)
    tiff_name = os.path.splitext(tiff_filename)[0]
    
    # 构建输出目录：OUTPUT_ROOT/s1/a01
    output_dir = os.path.join(output_root, subfolder, tiff_name)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 读取TIFF文件（支持3D z-stack数据）
        tiff_data = tiff.imread(tiff_path)
        
        # 确保数据是3D（z, h, w），如果是2D则扩展为3D（兼容单切片TIFF）
        if len(tiff_data.shape) == 2:
            tiff_data = np.expand_dims(tiff_data, axis=0)
        
        # 步骤1：全局归一化（除以TIFF最大值）
        normalized_data = normalize_tiff_data(tiff_data)
        
        # 逐切片处理：去噪 + 保存PNG
        num_slices = normalized_data.shape[0]
        for slice_idx in range(num_slices):
            # 提取单切片数据
            slice_data = normalized_data[slice_idx]
            
            # 步骤2：去噪滤波（核心新增步骤）
            denoised_slice = denoise_slice(slice_data, filter_type, kernel_size, sigmaX)
            
            # 切片索引从1开始命名（z1, z2...）
            slice_name = f"z{slice_idx + 1}.png"
            slice_path = os.path.join(output_dir, slice_name)
            
            # 步骤3：保存去噪后的PNG
            cv2.imwrite(slice_path, denoised_slice)
        
        return True, num_slices
    
    except Exception as e:
        print(f"\n❌ 处理文件失败 {tiff_path}：{str(e)}")
        return False, 0

def main():
    # ---------------------- 配置参数（按需修改） ----------------------
    INPUT_ROOT = "data_BBBC006/tif"    # 输入根目录（含s1/s2子文件夹）
    OUTPUT_ROOT = "data_BBBC006/images"# 输出根目录
    # 滤波配置（可根据噪声类型调整）
    FILTER_TYPE = "gaussian"  # 可选：gaussian（高斯）/median（中值）/none（无滤波）
    KERNEL_SIZE = 3           # 滤波核大小（奇数，3/5/7，越大去噪越强）
    SIGMA_X = 1.0             # 高斯滤波标准差（仅FILTER_TYPE=gaussian时生效）
    # ------------------------------------------------------------------

    # 验证输入目录是否存在
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 输入目录不存在：{INPUT_ROOT}")
        return
    
    # 创建输出根目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 定义需要处理的子文件夹（s1、s2）
    target_subfolders = ["s1", "s2"]
    
    # 统计总处理信息
    total_files = 0
    success_files = 0
    total_slices = 0
    
    # 打印滤波配置信息
    print(f"📌 滤波配置：类型={FILTER_TYPE}，核大小={KERNEL_SIZE}，高斯标准差={SIGMA_X}")
    
    # 遍历目标子文件夹
    for subfolder in target_subfolders:
        subfolder_path = os.path.join(INPUT_ROOT, subfolder)
        
        # 检查子文件夹是否存在
        if not os.path.exists(subfolder_path):
            print(f"⚠️  子文件夹不存在，跳过：{subfolder_path}")
            continue
        
        # 获取子文件夹下所有TIFF文件
        tiff_files = [
            os.path.join(subfolder_path, f) 
            for f in os.listdir(subfolder_path)
            if f.lower().endswith((".tif", ".tiff"))
        ]
        
        if not tiff_files:
            print(f"⚠️  子文件夹 {subfolder} 中未找到TIFF文件")
            continue
        
        # 打印当前子文件夹处理信息
        print(f"\n📌 开始处理子文件夹 {subfolder}，共找到 {len(tiff_files)} 个TIFF文件")
        
        # 遍历并处理每个TIFF文件（带进度条）
        for tiff_file in tqdm(tiff_files, desc=f"处理 {subfolder}", unit="文件"):
            total_files += 1
            # 处理单个TIFF文件（传入滤波参数）
            success, slices = process_tiff_to_png(
                tiff_file, OUTPUT_ROOT, subfolder,
                filter_type=FILTER_TYPE,
                kernel_size=KERNEL_SIZE,
                sigmaX=SIGMA_X
            )
            if success:
                success_files += 1
                total_slices += slices
    
    # 输出最终统计结果
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