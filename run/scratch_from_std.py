import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os


def scratch_from_region(tif_path, output_path, n_col, n_row, show_label=False, gamma=1.0):
    """
    从 TIF 的每个 z-slice 分块，逐块选 sharpness 最大的 z 层，拼接为最终 PNG。

    sharpness[z] = std(sub_img[:,:,z]) / max(sub_img[:,:,z])

    gamma: gamma 校正系数，>1 缩小高低像素差距，=1 不改变
    """
    tif = Image.open(tif_path)
    z_slices = []
    for i in range(tif.n_frames):
        tif.seek(i)
        z_slices.append(np.array(tif))
    volume = np.stack(z_slices, axis=2)  # (H, W, Z)

    H, W, Z = volume.shape
    block_h = H // n_row
    block_w = W // n_col

    result = np.zeros((H, W), dtype=volume.dtype)
    best_z_map = np.zeros((n_row, n_col), dtype=int)

    for row in range(n_row):
        for col in range(n_col):
            y_start = row * block_h
            y_end = (row + 1) * block_h if row < n_row - 1 else H
            x_start = col * block_w
            x_end = (col + 1) * block_w if col < n_col - 1 else W

            sub_vol = volume[y_start:y_end, x_start:x_end, :].astype(np.float64)

            sharpness = np.zeros(Z)
            for z in range(Z):
                sub = sub_vol[:, :, z]
                sharpness[z] = np.std(sub) / np.mean(sub)

            best_z = int(np.argmax(sharpness))
            result[y_start:y_end, x_start:x_end] = volume[y_start:y_end, x_start:x_end, best_z]

            best_z_map[row, col] = best_z

    # gamma 校正
    result_float = result.astype(np.float64)
    max_val = result_float.max()
    if max_val > 0 and gamma != 1.0:
        result_float = ((result_float / max_val) ** (1.0 / gamma) * 255)
    result = np.clip(result_float, 0, 255).astype(np.uint8)

    Image.fromarray(result).save(output_path)

    if show_label:
        result_img = Image.fromarray(result)
        draw = ImageDraw.Draw(result_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        for row in range(n_row):
            for col in range(n_col):
                y_start = row * block_h
                x_start = col * block_w
                draw.text((x_start + 2, y_start + 2), f"z{best_z_map[row, col]}",
                          fill=255, font=font, stroke_width=2, stroke_fill=0)
        result_img.save(output_path)
        print(f"已标注 z 层号，保存到：{output_path}")
    else:
        print(f"保存到：{output_path}")


if __name__ == "__main__":

    data_type = "brain"
    image_idx = '13'

    parser = argparse.ArgumentParser(description="分块选最优z层拼接TIF")
    parser.add_argument("--tif", type=str, default=f"outputs/{data_type}/selected/{image_idx}/pred_z8_z12.tif", help="输入 TIF 路径")
    parser.add_argument("--output", type=str, default=f"outputs/{data_type}/selected/{image_idx}/pred_sharpness.png", help="输出 PNG 路径")
    parser.add_argument("--n_col", type=int, default=8, help="x 方向分块数")
    parser.add_argument("--n_row", type=int, default=8, help="y 方向分块数")
    parser.add_argument("--show_label", action="store_true", default=False, help="在每个子图左上角标注 z 层号")
    parser.add_argument("--gamma", type=float, default=1.25, help="gamma 校正系数，>1 缩小高低像素差距")
    args = parser.parse_args()

    scratch_from_region(args.tif, args.output, args.n_col, args.n_row, args.show_label, args.gamma)