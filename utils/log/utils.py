import torch
from torchvision.utils import make_grid



def make_grid_for_multichannel(images, nrow=4):
    """保持原有多通道图像网格生成逻辑"""
    batch_size, channels, height, width = images.shape
    
    if channels == 1:
        return make_grid(images, nrow=nrow)
    
    reshaped = images.view(batch_size * channels, 1, height, width)
    grid = make_grid(reshaped, nrow=channels*nrow//4 if channels*nrow//4 > 0 else 1)
    
    return grid

def normalize_tensor(tensor):
    """保持原有张量归一化逻辑"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)

def normalize_images(pred, gt, lq):
    """保持原有图像归一化逻辑"""
    pred_norm = torch.stack([normalize_tensor(img) for img in pred])
    gt_norm = torch.stack([normalize_tensor(img) for img in gt])
    lq_norm = torch.stack([normalize_tensor(img) for img in lq])
    return pred_norm, gt_norm, lq_norm

def log_images(writer, pred, pred_norm, gt, gt_norm, lq, lq_norm, global_step):
    """保持原有TensorBoard图像日志逻辑"""
    for tag, image in [
        ("image/pred", pred),
        ("image/pred_normalized", pred_norm),
        ("image/gt", gt),
        ("image/gt_normalized", gt_norm),
        ("image/lq", lq),
        ("image/lq_normalized", lq_norm),
    ]:
        if tag == "image/lq":
            writer.add_image(tag, make_grid_for_multichannel(image, nrow=4), global_step)
        else:
            writer.add_image(tag, make_grid(image, nrow=4), global_step)