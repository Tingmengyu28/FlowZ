import numpy as np
import tifffile as tif
import cv2
import os
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings('ignore')

def load_tiff_stack(tiff_path):
    with tif.TiffFile(tiff_path) as tif_file:
        stack = tif_file.asarray()
    stack = np.transpose(stack, (1, 2, 0))
    print(f"成功加载图像堆栈，shape: {stack.shape}")
    return stack

def generate_edf_image(stack):
    edf = np.max(stack, axis=2)
    return edf.astype(np.float32)

def get_roi_mask(edf_image):
    thresh = threshold_triangle(edf_image)
    binary_mask = edf_image > thresh
    selem = generate_binary_structure(2, 2)
    dilated_mask = binary_dilation(binary_mask, structure=selem, iterations=3)
    labeled_mask = label(dilated_mask)
    regions = regionprops(labeled_mask)
    valid_regions = [r.label for r in regions if r.area > 100]
    roi_mask = np.isin(labeled_mask, valid_regions)
    print(f"ROI掩码生成完成，有效信号区域占比: {np.sum(roi_mask)/roi_mask.size:.2%}")
    return roi_mask

def greedy_crop_image_patches(stack, roi_mask, patch_size=256, overlap_ratio=0.05):
    height, width, depth = stack.shape
    patch_s = patch_size
    overlap = int(patch_s * overlap_ratio)
    step = patch_s - overlap
    covered_mask = np.zeros_like(roi_mask, dtype=bool)
    patches = []
    patch_coords = []
    y_centers = np.arange(patch_s//2, height, step, dtype=int)
    x_centers = np.arange(patch_s//2, width, step, dtype=int)
    while np.any(roi_mask & (~covered_mask)):
        remaining_roi = roi_mask & (~covered_mask)
        if not np.any(remaining_roi):
            break
        max_cov = 0
        best_y, best_x = patch_s//2, patch_s//2
        for yc in y_centers:
            for xc in x_centers:
                y1 = max(0, yc - patch_s//2)
                y2 = min(height, yc + patch_s//2)
                x1 = max(0, xc - patch_s//2)
                x2 = min(width, xc + patch_s//2)
                if y2 - y1 != patch_s or x2 - x1 != patch_s:
                    continue
                cov = np.sum(remaining_roi[y1:y2, x1:x2])
                if cov > max_cov and cov > 0:
                    max_cov = cov
                    best_y, best_x = yc, xc
        y1 = best_y - patch_s//2
        y2 = best_y + patch_s//2
        x1 = best_x - patch_s//2
        x2 = best_x + patch_s//2
        patch = stack[y1:y2, x1:x2, :]
        c_y1 = max(0, y1 + overlap)
        c_y2 = min(height, y2 - overlap)
        c_x1 = max(0, x1 + overlap)
        c_x2 = min(width, x2 - overlap)
        covered_mask[c_y1:c_y2, c_x1:c_x2] = True
        patches.append(patch)
        patch_coords.append((y1, y2, x1, x2))
        print(f"裁剪图像块 {len(patches)}: 坐标 ({y1}:{y2}, {x1}:{x2})")
        if np.sum(roi_mask & (~covered_mask)) == np.sum(remaining_roi):
            break
    patches = np.array(patches)
    print(f"\n贪心裁剪完成，共得到 {len(patches)} 个{patch_s}×{patch_s}图像块")
    return patches, patch_coords

def normalize_patches(patches):
    normalized_patches = []
    for patch in patches:
        patch_edf = np.max(patch, axis=2)
        bg = np.mean(patch_edf[patch_edf < np.percentile(patch_edf, 10)])
        patch_norm = patch - bg
        max_i = np.percentile(patch_norm[patch_norm > 0], 99)
        if max_i > 0:
            patch_norm = patch_norm / max_i
        patch_norm = np.clip(patch_norm, 0, 1)
        normalized_patches.append(patch_norm)
    return np.array(normalized_patches)

# 核心：替换原高斯滤波，默认双边滤波（保边），可切换非局部均值（强去噪）
def denoise_patches(patches, method='bilateral', ksize=(3,3)):
    denoised_patches = []
    for patch in patches:
        denoised_patch = []
        for z in range(patch.shape[2]):
            slice_8bit = (patch[:, :, z] * 255).astype(np.uint8)
            if method == 'bilateral':
                # 双边滤波：d=邻域大小，sigmaColor=像素值相似度，sigmaSpace=空间距离
                denoised_slice = cv2.bilateralFilter(slice_8bit, d=ksize[0], sigmaColor=75, sigmaSpace=75)
            elif method == 'nlmeans':
                # 非局部均值：h=去噪强度（越大去噪越强，默认10适配高斯噪声）
                denoised_slice = cv2.fastNlMeansDenoising(slice_8bit, h=10)
            else:
                # 兼容原高斯滤波（备用）
                denoised_slice = cv2.GaussianBlur(slice_8bit, ksize, 0)
            denoised_patch.append(denoised_slice)
        denoised_patches.append(np.stack(denoised_patch, axis=2))
    return np.array(denoised_patches)

def save_patches(patches, output_dir, start_idx):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(patches)):
        sub_dir = os.path.join(output_dir, str(start_idx + i))
        os.makedirs(sub_dir, exist_ok=True)
        for z in range(patches[i].shape[2]):
            cv2.imwrite(os.path.join(sub_dir, f"z{z+1}.png"), patches[i][:, :, z])
    return start_idx + len(patches)

def main(tiff_path, start_idx):
    stack = load_tiff_stack(tiff_path)
    edf = generate_edf_image(stack)
    roi = get_roi_mask(edf)
    patches, coords = greedy_crop_image_patches(stack, roi)
    patches_norm = normalize_patches(patches)
    # 去噪：method='bilateral'（保边）/ 'nlmeans'（强去噪）/ 'gaussian'（原方法）
    patches_denoised = denoise_patches(patches_norm, method='gaussian')
    next_idx = save_patches(patches_denoised, output_dir=f"data/images_cropped/ch{ch}", start_idx=start_idx)
    print(f"\n当前文件处理完成，下一个起始序号: {next_idx}")
    print(f"- 本次生成图像块数量: {len(patches)}")
    print(f"- 单个图像块尺寸: {patches[0].shape}")
    return next_idx

if __name__ == "__main__":
    global_idx = 1
    ch = 2
    for i in range(1, 13):
        if 1 <= i <= 6:
            TIFF_FILE_PATH = f"data/tif/WF_merged/Fixed Hela WF-{i:02d}/Fixed Hela WF-{i:02d}_ch{ch:02d}.tif"
        elif 7 <= i <= 10:
            TIFF_FILE_PATH = f"data/tif/WF_merged/Fixed Hela WF-{i:02d}/Project_Fixed Hela WF-{i:02d}_z00_ch{ch:02d}.tif"
        else:
            TIFF_FILE_PATH = f"data/tif/WF_merged/Fixed Hela WF-{i:02d}/Project001_Fixed Hela WF-{i:02d}_ch{ch:02d}.tif"
        
        try:
            global_idx = main(TIFF_FILE_PATH, global_idx)
        except Exception as e:
            print(f"文件 {i:02d} 执行出错: {str(e)}")
            continue
    print(f"最终生成图像块文件夹序号: 1 ~ {global_idx-1}")