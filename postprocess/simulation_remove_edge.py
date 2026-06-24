import os
import numpy as np
import cv2

def apply_pixel_threshold(image_path, output_path, threshold):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    
    img_filtered = img.copy()
    mask = img_filtered < threshold
    img_filtered[mask] = 0
    
    cv2.imwrite(output_path, img_filtered)
    zero_pixels = np.sum(mask)
    total_pixels = img.size
    print(f"处理完成 - {os.path.basename(image_path)}")
    print(f"阈值: {threshold}, 置0像素数: {zero_pixels}/{total_pixels} ({zero_pixels/total_pixels*100:.2f}%)")
    print(f"输出文件: {output_path}\n")

def main():
    pred_path = "outputs/simulation/fm_palette/inference_z_stacks/pred.png"
    gt_path = "outputs/simulation/fm_palette/inference_z_stacks/gt.png"
    threshold_value = 120
    output_dir = "outputs/simulation/fm_palette/inference_z_stacks/thresholded"
    
    os.makedirs(output_dir, exist_ok=True)
    
    pred_output = os.path.join(output_dir, "pred_thresholded.png")
    gt_output = os.path.join(output_dir, "gt_thresholded.png")
    
    try:
        apply_pixel_threshold(pred_path, pred_output, threshold_value)
        apply_pixel_threshold(gt_path, gt_output, threshold_value)
        print("所有图像处理完成！")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()