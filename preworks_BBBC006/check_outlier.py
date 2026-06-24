import os
# import random
from PIL import Image
import math

def create_image_grid_from_subfolders():
    # ====================== 配置参数 ======================
    # 源目录：需要遍历的根目录（包含所有子文件夹）
    source_root = "data_BBBC006/images/s1"
    # 网格图保存路径（可自行修改，后缀建议用.png）
    grid_save_path = "image_grid.png"
    # 每个小图片在网格中的统一尺寸（宽x高，可根据需求调整）
    img_size = (256, 256)
    # 网格中图片之间的间距（像素）
    padding = 5

    # ====================== 步骤1：收集所有选中的PNG文件路径 ======================
    selected_image_paths = []
    subfolder_names = []  # 记录子文件夹名，可选：用于标注或验证

    # 遍历源目录下的所有子文件夹
    for subfolder_name in os.listdir(source_root):
        subfolder_path = os.path.join(source_root, subfolder_name)
        
        # 只处理文件夹（跳过文件）
        if not os.path.isdir(subfolder_path):
            continue
        
        # 筛选出当前子文件夹中的所有PNG文件
        png_files = []
        for file_name in os.listdir(subfolder_path):
            if file_name.lower().endswith(".png"):
                png_files.append(os.path.join(subfolder_path, file_name))
        
        # 跳过没有PNG文件的子文件夹
        if not png_files:
            print(f"警告：子文件夹 {subfolder_name} 中未找到PNG文件，已跳过")
            continue
        
        # 随机选择一个PNG文件
        selected_file = png_files[len(png_files)//2]
        selected_image_paths.append(selected_file)
        subfolder_names.append(subfolder_name)
        print(f"已选中 {subfolder_name} 文件夹的文件：{os.path.basename(selected_file)}")

    # 无有效图片时直接退出
    if not selected_image_paths:
        print("错误：未找到任何可处理的PNG文件！")
        return

    # ====================== 步骤2：读取并调整所有图片尺寸 ======================
    images = []
    for img_path in selected_image_paths:
        try:
            # 打开图片并转为RGB（避免透明通道干扰）
            img = Image.open(img_path).convert("RGB")
            # 调整尺寸（保持比例+裁剪，或直接拉伸，这里用resize简单处理）
            img_resized = img.resize(img_size, Image.Resampling.LANCZOS)
            images.append(img_resized)
        except Exception as e:
            print(f"警告：读取图片 {img_path} 失败，已跳过，原因：{str(e)}")
            # 移除对应文件夹名，保持列表长度一致
            idx = selected_image_paths.index(img_path)
            subfolder_names.pop(idx)
            selected_image_paths.pop(idx)

    # ====================== 步骤3：计算网格行列数（接近正方形） ======================
    num_imgs = len(images)
    # 计算行数（向上取整平方根）
    rows = math.ceil(math.sqrt(num_imgs))
    # 计算列数（向上取整：总图片数/行数）
    cols = math.ceil(num_imgs / rows)

    # ====================== 步骤4：创建网格画布并拼接图片 ======================
    # 计算画布总尺寸：(列数*单图宽 + (列数-1)*间距, 行数*单图高 + (行数-1)*间距)
    canvas_width = cols * img_size[0] + (cols - 1) * padding
    canvas_height = rows * img_size[1] + (rows - 1) * padding
    # 创建白色背景的画布
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # 逐个将图片粘贴到画布对应位置
    for idx, img in enumerate(images):
        # 计算当前图片的行列位置
        row_idx = idx // cols
        col_idx = idx % cols
        # 计算粘贴的左上角坐标
        x = col_idx * (img_size[0] + padding)
        y = row_idx * (img_size[1] + padding)
        # 粘贴图片
        canvas.paste(img, (x, y))

    # ====================== 步骤5：保存网格图 ======================
    try:
        canvas.save(grid_save_path)
        print(f"\n成功！网格图已保存至：{os.path.abspath(grid_save_path)}")
        print(f"网格规格：{rows} 行 × {cols} 列，共 {num_imgs} 张图片")
    except Exception as e:
        print(f"错误：保存网格图失败，原因：{str(e)}")

if __name__ == "__main__":
    create_image_grid_from_subfolders()