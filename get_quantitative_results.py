import cv2
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim


def calculate_metrics(img1_path, img2_path):
    """计算两张图像的 Pearson 相关系数和 SSIM"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        raise FileNotFoundError(f"无法读取图像: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"无法读取图像: {img2_path}")

    if img1.shape != img2.shape:
        raise ValueError(f"图像尺寸不匹配: {img1.shape} vs {img2.shape}")

    # Pearson 相关系数
    flat1 = img1.flatten().astype(np.float64)
    flat2 = img2.flatten().astype(np.float64)
    pearson_corr, _ = pearsonr(flat1, flat2)

    # SSIM
    ssim_val = ssim(img1, img2, data_range=255)

    return pearson_corr, ssim_val


if __name__ == "__main__":
    img1_path = "outputs/real_slope/selected/70/input_raw.png"  # 修改为实际路径
    img2_path = "outputs/real_slope/selected/70/input_raw.png"  # 修改为实际路径

    pearson_corr, ssim_val = calculate_metrics(img1_path, img2_path)

    print(f"Pearson 相关系数: {pearson_corr:.6f}")
    print(f"SSIM: {ssim_val:.6f}")
