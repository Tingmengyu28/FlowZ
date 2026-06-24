import numpy as np
from PIL import Image
import tifffile
import os

def process_tif_with_max_pooling(tif_path, selected_layer_index, L_min, L_max, output_dir=None):
    """
    读取tif文件并对指定范围的层进行max pooling
    
    Args:
        tif_path: tif文件路径
        selected_layer_index: 选定的层的序号
        image_idx: 图像索引，用于生成输出文件名
        L_min: 范围参数，处理[selected_layer_index-L_min, selected_layer_index+L_max]范围内的层
        L_max: 范围参数，处理[selected_layer_index-L_min, selected_layer_index+L_max]范围内的层
        output_dir: 输出目录，默认为tif文件所在目录
    """
    # 读取tif文件
    tif_data = tifffile.imread(tif_path)
    
    # 获取tif文件信息
    print(f"原始tif数据形状: {tif_data.shape}")
    
    # 如果是单层图像，扩展维度
    if len(tif_data.shape) == 2:
        tif_data = tif_data[np.newaxis, :, :]
    
    # 计算归一化因子（整个tif的最大值）
    max_value = np.max(tif_data)
    print(f"整个tif的最大值: {max_value}")
    
    # 归一化整个tif数据
    normalized_data = tif_data / max_value
    
    # 计算实际的起始和结束层索引，确保不超出边界
    start_layer_positive = min(normalized_data.shape[0] - 1, selected_layer_index + L_min)
    end_layer_positive = min(normalized_data.shape[0] - 1, selected_layer_index + L_max)
    start_layer_negative = max(0, selected_layer_index - L_max)
    end_layer_negative = max(0, selected_layer_index - L_min)
    
    # 提取指定范围的层
    selected_layers_positive = normalized_data[start_layer_positive:end_layer_positive+1]
    selected_layers_negative = normalized_data[start_layer_negative:end_layer_negative+1]
    selected_layers = np.concatenate((selected_layers_negative, selected_layers_positive), axis=0)

    # 对选定范围的层进行max pooling（在轴0上取最大值，即沿层数方向）
    max_pooled_image = np.max(selected_layers, axis=0)
    print(f"Max pooling后图像形状: {max_pooled_image.shape}")
    
    # 将归一化的图像转换为8位图像用于保存
    max_pooled_image_uint8 = (max_pooled_image * 255).astype(np.uint8)
    
    # 设置输出路径
    if output_dir is None:
        output_dir = os.path.dirname(tif_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    output_path = os.path.join(output_dir, "gt.png")
    
    # 保存图像
    Image.fromarray(max_pooled_image_uint8).save(output_path)
    print(f"Max pooling结果已保存至: {output_path}")
    
    return max_pooled_image_uint8

# 示例使用
if __name__ == "__main__":
    # 参数设置
    image_idx = "09"
    selected_layer_index = 45
    L_min, L_max = 5, 25
    ch = 2
    model_name = "fm_palette_L25"

    tif_path = f"/data1/azt/cv/recoverZ/data/tif/CFC_merged/Fixed Hela Con-{image_idx}/Project_Fixed Hela Con-{image_idx}_z00_ch0{ch}.tif"
    output_dir = f"/data1/azt/cv/recoverZ/outputs/inference/{model_name}/ch{ch}/{image_idx}/"
    
    # 执行处理
    result = process_tif_with_max_pooling(tif_path, selected_layer_index, L_min, L_max, output_dir)
