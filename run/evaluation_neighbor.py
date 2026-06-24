import pandas as pd
import numpy as np
import os


def load_points(file_path):
    """加载点数据文件"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)  # 添加错误行跳过参数
    elif file_path.endswith('.txt'):
        return pd.read_csv(file_path, sep=' ', header=None, names=['frame', 'x', 'y', 'z'])
    else:
        raise ValueError("Unsupported file format")

def find_neighbors(point, points, visited, threshold=(2,2,4)):
    """深度优先搜索查找相邻点"""
    neighbors = []
    stack = [point]
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
            
        visited.add(current)
        neighbors.append(current)
        
        # 将numpy数组转换为元组列表
        tuple_points = [tuple(p) for p in points]

        # 查找三维空间中的相邻点（曼哈顿距离≤1）
        for p in tuple_points:
            if (abs(p[0]-current[0]) <= threshold[0] and 
                abs(p[1]-current[1]) <= threshold[1] and
                abs(p[2]-current[2]) <= threshold[2]):
                if p not in visited:
                    stack.append(p)
    
    return neighbors

def cluster_points(pred_df, target_frame):
    """主聚类函数"""
    clustered = []
    
    # 按frame分组处理
    for frame, group in pred_df.groupby('frame'):
        # 添加frame过滤逻辑
        if target_frame is not None and frame != target_frame:
            continue
        coords = group[['x', 'y', 'z']].values
        intensities = group['intensity'].values
        
        # 按强度降序排序
        sorted_indices = np.argsort(-intensities)
        remaining_points = coords[sorted_indices]
        remaining_intensities = intensities[sorted_indices]
        
        visited = set()
        clusters = []
        
        # 迭代处理每个点
        for i in range(len(remaining_points)):
            point = tuple(remaining_points[i])
            if point not in visited:
                # 查找连通区域
                cluster_points = find_neighbors(point, remaining_points, visited)
                # cluster_points = remaining_points[cluster_indices]
                cluster_indices = []
                tuple_remaining = [tuple(p) for p in remaining_points]
                for p in cluster_points:
                    try:
                        idx = tuple_remaining.index(tuple(p))
                        cluster_indices.append(idx)
                    except ValueError:
                        pass  # 忽略不存在的情况
                cluster_intensities = remaining_intensities[cluster_indices]
                
                # 计算质心
                total_intensity = cluster_intensities.sum()
                centroid = np.average(cluster_points, axis=0, 
                                    weights=cluster_intensities)
                
                clusters.append({
                    'frame': frame,
                    'x': centroid[0],
                    'y': centroid[1],
                    'z': centroid[2],
                    'intensity': total_intensity
                })
        
        clustered.extend(clusters)
    
    return pd.DataFrame(clustered)


def calculate_metrics(clustered_pred, target_points, distance_threshold=[2,2,4]):
    """计算召回率和精确率"""
    metrics = {
        'frame': [],
        'precision': [],
        'recall': [],
        'true_positives': [],
        'predicted_points': [],
        'ground_truth_points': []
    }
    
    # 遍历所有frame
    for frame_id in clustered_pred['frame'].unique():
        # 获取预测点
        pred = clustered_pred[clustered_pred['frame'] == frame_id]
        # 获取真实点
        gt = target_points[target_points['frame'] == frame_id]
        
        # 转换坐标格式
        pred_points = pred[['x', 'y', 'z']].values
        gt_points = gt[['x', 'y', 'z']].values
        
        # 匹配预测和真实点（曼哈顿距离≤阈值）
        tp = 0
        matched_gt = set()
        
        for p in pred_points:
            for i, g in enumerate(gt_points):
                if (abs(p[0]-g[0]) <= distance_threshold[0] and 
                    abs(p[1]-g[1]) <= distance_threshold[1] and
                    abs(p[2]-g[2]) <= distance_threshold[2]):
                    if i not in matched_gt:
                        tp += 1
                        matched_gt.add(i)
                        break
        
        # 计算指标
        precision = tp / len(pred) if len(pred) > 0 else 0
        recall = tp / len(gt) if len(gt) > 0 else 0
        
        # 记录结果
        metrics['frame'].append(frame_id)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['true_positives'].append(tp)
        metrics['predicted_points'].append(len(pred))
        metrics['ground_truth_points'].append(len(gt))
    
    return pd.DataFrame(metrics)


def main():
    out_dir = 'outputs/simulation/fm_palette/inference_results'
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    pred_points = load_points('outputs/simulation/fm_palette/inference_z_stacks/loc.csv')
    target_points = load_points('data_simulation/points_coordinates.txt')

    with open(os.path.join(out_dir, 'eval.txt'), 'w') as f:
        f.write("Evaluation Results\n")
        f.write("==================\n\n")
    
    # Get sorted frame list
    frame_ids = sorted(target_points['frame'].unique())
    all_metrics = []
    csv_data = {
        'Points': [],
        'Precision': [],
        'Recall': []
    }
    
    # Process in batches of 100 frames
    for i in range(0, len(frame_ids), 100):
        batch = frame_ids[i:i+100]
        start_id = batch[0]
        end_id = batch[-1]
        num_points = 5 * (start_id//100 + 1)
        batch_metrics = []
        for frame_id in batch:
            clustered_pred = cluster_points(pred_points, target_frame=frame_id)
            frame_metrics = calculate_metrics(clustered_pred, target_points, distance_threshold=[2,2,4])
            batch_metrics.append(frame_metrics)
        
        combined = pd.concat(batch_metrics)
        avg_precision = round(combined['precision'].mean(), 4)
        avg_recall = round(combined['recall'].mean(), 4)
        csv_data['Points'].append(num_points)
        csv_data['Precision'].append(avg_precision)
        csv_data['Recall'].append(avg_recall)
        
        with open('out/eval.txt', 'a') as f:
            print(f"Frames {start_id}-{end_id} with {5*(start_id//100+1)} Points:")
            f.write(f"Frames {start_id}-{end_id} with {5*(start_id//100+1)} Points:\n")
            print(f"Average Precision: {avg_precision:.2%}")
            f.write(f"Average Precision: {avg_precision:.2%}\n")
            print(f"Average Recall: {avg_recall:.2%}\n")
            f.write(f"Average Recall: {avg_recall:.2%}\n\n")
        all_metrics.extend(batch_metrics)
    
    # Final statistics
    final_metrics = pd.concat(all_metrics)
    with open('out/eval.txt', 'a') as f:
        print("\nGlobal Statistics:")
        f.write("\nGlobal Statistics:\n")
        print(f"Overall Average Precision: {final_metrics['precision'].mean():.2%}")
        f.write(f"Overall Average Precision: {final_metrics['precision'].mean():.2%}\n")
        print(f"Overall Average Recall: {final_metrics['recall'].mean():.2%}")
        f.write(f"Overall Average Recall: {final_metrics['recall'].mean():.2%}\n")

    # 添加总体平均值
    csv_data['Points'].append('Overall')
    csv_data['Precision'].append(final_metrics['precision'].mean())
    csv_data['Recall'].append(final_metrics['recall'].mean())
    
    pd.DataFrame(csv_data).to_csv(os.path.join(out_dir, 'eval.csv'), index=False)


if __name__ == "__main__":
    main()
