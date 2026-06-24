import numpy as np
from PIL import Image
import argparse
import os


def scratch_from_maxpool(tif_path, output_path, threshold_ratio=0.5, gamma=1.0):
    """
    TIF 3D max projection + 阈值过滤 + gamma 校正。

    1. 逐像素 (x, y) 取所有 z 层中的最大值，组成 max projection png
    2. 将低于 threshold_ratio * max(png) 的像素置 0
    3. gamma 校正：output = (input/max)^(1/gamma) * 255，gamma>1 时缩小高低像素差距
    """
    tif = Image.open(tif_path)
    z_slices = []
    for i in range(tif.n_frames):
        tif.seek(i)
        z_slices.append(np.array(tif))
    volume = np.stack(z_slices, axis=2).astype(np.float64)  # (H, W, Z)

    max_proj = volume.max(axis=2)

    threshold = threshold_ratio * max_proj.max()
    max_proj[max_proj < threshold] = 0

    max_val = max_proj.max()
    if max_val > 0:
        max_proj = ((max_proj / max_val) ** (1.0 / gamma) * 255)

    max_proj = np.clip(max_proj, 0, 255).astype(np.uint8)
    Image.fromarray(max_proj).save(output_path)
    print(f"保存到：{output_path}  (阈值: {threshold:.1f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIF Max Projection + 阈值过滤")
    parser.add_argument("--tif", type=str, default="outputs/real_slope/selected/13/pred_z0_z20.tif", help="输入 TIF 路径")
    parser.add_argument("--output", type=str, default="outputs/real_slope/selected/13/pred_maxpool.png", help="输出 PNG 路径")
    parser.add_argument("--threshold", type=float, default=0.2, help="阈值系数，低于 threshold*max 的像素置 0")
    parser.add_argument("--gamma", type=float, default=2.0, help="gamma 校正系数，>1 缩小高低像素差距")
    args = parser.parse_args()

    scratch_from_maxpool(args.tif, args.output, args.threshold, args.gamma)