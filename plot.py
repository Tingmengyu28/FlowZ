import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects
from PIL import Image
import os

def save_publication_3d_enhanced(file_path, image_dir, target_image_id, save_name="3d_scatter.png"):
    # --- 1. 学术论文高对比度配置 ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 2.0,
        "mathtext.fontset": "stix"
    })

    # --- 2. 加载数据 ---
    if not os.path.exists(file_path): return
    data = np.loadtxt(file_path)
    mask = (data[:, 0] == target_image_id)
    pts = data[mask]
    if len(pts) == 0: return

    y_p, x_p, z_p = pts[:, 1], pts[:, 2], pts[:, 3]

    # --- 3. ICA Colormap 与底图处理 ---
    target_cmap = plt.cm.hot 

    png_path = os.path.join(image_dir, str(int(target_image_id)), "z0.png")
    img_colored = None
    if os.path.exists(png_path):
        img_raw = np.array(Image.open(png_path).convert('L')) / 255.0
        
        # 归一化处理
        vmax_bg = np.percentile(img_raw, 99.9) * 1.1
        img_raw = np.clip(img_raw / vmax_bg, 0, 1)
        
        # 使用 'hot' 映射应用颜色
        img_colored = target_cmap(img_raw)

    # --- 4. 绘图 ---
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # A. ICA 底图 (物理下沉)
    if img_colored is not None:
        h, w = img_raw.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        # 核心修正：将底图物理位置设为 Z = 0 (或更小)，确保它在几何上永远处于最低点
        ax.plot_surface(xx, yy, np.full_like(xx, 0.0), facecolors=img_colored, 
                        shade=False, antialiased=True, rstride=2, cstride=2, alpha=1.0)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)

    # B. 投影虚线
    # 虚线从 Z = -0.8 (略高于底图) 开始，连向 z_p
    for i in range(len(x_p)):
        ax.plot([x_p[i], x_p[i]], [y_p[i], y_p[i]], [0.0, z_p[i]], 
                color='#111111',       # 使用非常深的黑色作为边框
                linestyle='-',        # 边框也用虚线，保持图案一致性（也可以用'-'实线）
                linewidth=2.4,        # 粗度是白色核心线的两倍左右
                alpha=0.6,             # alpha 设稍低，避免边框过硬
                zorder=9)             # zorder 设低一级

        # 2. 绘制顶层：白色虚线，作为核心
        ax.plot([x_p[i], x_p[i]], [y_p[i], y_p[i]], [0.0, z_p[i]], 
                color='#FFFFFF',       # 纯白色核心
                linestyle='--',        # 标准虚线
                linewidth=1.2,        # 标准粗度
                alpha=0.9,             # alpha 设高一点，突出核心
                zorder=10)            # zorder 设高一级，确保盖在黑色线上

    # C. 散点 (最后调用以获得最高的绘制上下文)
    # 每一个散点手动设置更高的 zorder 属性（虽然在3D中不一定生效，但能防止某些后端的遮挡）
    scatter = ax.scatter(x_p, y_p, z_p, 
                        c=z_p,                
                        cmap='viridis',       
                        s=150,                
                        alpha=1.0, 
                        edgecolors='white', 
                        linewidth=1.0,        
                        depthshade=False) # 设为 False 后点不会随深度变灰，看起来更“前置”

    # --- 5. 坐标轴精修 ---
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.xaxis.line.set_lw(2.0)
    ax.yaxis.line.set_lw(2.0)
    ax.zaxis.line.set_lw(2.0)

    ax.set_xlabel('X Coordinate', labelpad=15)
    ax.set_ylabel('Y Coordinate', labelpad=15)
    ax.set_zlim(0, max(z_p.max(), 20)) # 调整 Z 轴范围，给底图留出空间

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    cbar.set_label('Measured Z Coordinate', weight='bold')

    ax.view_init(elev=25, azim=38)
    ax.set_box_aspect((1, 1, 0.6)) 

    plt.tight_layout()
    plt.savefig(save_name, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Publication-ready plot saved: {save_name}")

if __name__ == "__main__":
    save_publication_3d_enhanced('data_simulation/points_coordinates.txt', 
                                 'data_simulation/images', 1)