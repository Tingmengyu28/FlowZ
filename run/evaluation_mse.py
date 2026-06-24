import os
import cv2
import numpy as np
from glob import glob

def calculate_mse_for_depth_maps(gt_dir, pred_dir):
    """
    计算两个目录中相同名称PNG图像的MSE
    """
    gt_files = glob(os.path.join(gt_dir, "*.png"))
    
    if not gt_files:
        print(f"在 {gt_dir} 中未找到PNG文件")
        return None
    
    gt_imgs, pred_imgs = [], []
    
    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, filename)
        
        if not os.path.exists(pred_file):
            print(f"警告: 预测文件不存在 {pred_file}")
            continue
        
        gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        
        if gt_img.shape != pred_img.shape:
            print(f"图像尺寸不匹配 {gt_file} 和 {pred_file}: {gt_img.shape} vs {pred_img.shape}")
            continue
        
        gt_imgs.append(gt_img)
        pred_imgs.append(pred_img)
        
    gt_imgs = np.stack(gt_imgs, axis=0)
    pred_imgs = np.stack(pred_imgs, axis=0)
        
    both_nonzero_mask = (gt_imgs != 0) & (pred_imgs != 0)
    pred_nonzero_mask = (pred_imgs != 0)
    gt_nonzero_mask = (gt_imgs != 0)
    both_zero_mask = (gt_imgs == 0) & (pred_imgs == 0)
    
    total_pixels = gt_imgs.size
    consistent_pixels = np.sum(both_nonzero_mask)
    consistency_ratio = consistent_pixels / (total_pixels - np.sum(both_zero_mask))
    recall_ratio = np.sum(both_nonzero_mask) / np.sum(gt_nonzero_mask)
    precision_ratio = np.sum(both_nonzero_mask) / np.sum(pred_nonzero_mask)
    
    gt_valid = gt_imgs[both_nonzero_mask].astype(np.float32)
    pred_valid = pred_imgs[both_nonzero_mask].astype(np.float32)
        
    rmse = np.mean((gt_valid - pred_valid) ** 2) ** 0.5 / 10
    
    print(f"文件 {filename}: RMSE = {rmse:.6f} (基于 {np.sum(both_nonzero_mask)} 个GT和预测图像都非零的像素)")
    print(f" Accuracy: {consistency_ratio:.6f}")
    print(f" Recall: {recall_ratio:.6f}")
    print(f" Precision: {precision_ratio:.6f}")
    
    overall_rmse = np.mean(rmse)
    
    print(f"\n总共有 {gt_imgs.shape[0]} 对图像")
    print(f"整体RMSE均值: {overall_rmse:.6f}")
    
    return overall_rmse

if __name__ == "__main__":
    gt_dir = "outputs/simulation/fm_palette/inference_details/depth_gt"
    pred_dir = "outputs/simulation/fm_palette/inference_details/depth_pred"
    
    mse_mean = calculate_mse_for_depth_maps(gt_dir, pred_dir)
    
    if mse_mean is not None:
        print(f"\n完成! 深度图的整体MSE均值为: {mse_mean:.6f}")
    else:
        print("\n计算失败!")