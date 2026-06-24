import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.common import to, calculate_psnr_pt


def sample_time_steps(batch_size, device):
    return torch.sigmoid(torch.rand(batch_size, device=device))


def drop_labels(cond, label_drop_prob=0.1):
    batch_size = cond.shape[0]
    device = cond.device
    drop_mask = torch.rand(batch_size, device=device) < label_drop_prob
    drop_mask_expanded = drop_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    uncondition_cond = torch.zeros_like(cond, device=device, requires_grad=False)
    out = torch.where(drop_mask_expanded, uncondition_cond, cond)
    
    return out


def get_flow_matching_x_t(x0, z, t):
    """
    生成t时刻的插值样本x_t（线性插值，CFM简化版）
    Args:
        x0: (B, C, H, W)，目标样本（gt）
        z: (B, C, H, W)，高斯噪声（初始分布样本）
        t: (B,)，时间步
    Returns:
        x_t: (B, C, H, W)，t时刻的插值样本
    """
    # 扩展t的维度，适配图像形状：(B,) -> (B, 1, 1, 1)
    t_expand = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    x_t = (1.0 - t_expand) * z + t_expand * x0
    return x_t


def get_flow_matching_target_velocity(x0, x_t, t):
    """
    计算Flow Matching的目标速度场v_t^target（CFM简化版）
    修正：扩展t的维度以支持广播运算
    Args:
        x0: (B, C, H, W)，目标样本（gt）
        x_t: (B, C, H, W)，初始分布样本
        t: (B,)，时间步
    Returns:
        v_t_target: (B, C, H, W)，目标速度场
    """
    B = t.shape[0]
    t_unsqueezed = t.reshape(B, 1, 1, 1)
    return (x0 - x_t) / (1.0 - t_unsqueezed).clamp_min(5e-2)


def flow_matching_forward_pass(model, x0, cond, device):
    """
    Flow Matching单次前向传播（完整训练步骤）
    已修复维度不匹配错误：扩展时间步t的维度以支持广播运算
    Args:
        model: 模型实例
        x0: (B, C, H, W)，目标样本（gt）
        cond: (B, C_cond, H, W)，条件输入（lq + dpm）
        device: 计算设备
    Returns:
        v_t_pred: 预测速度场，v_t_target: 目标速度场，x_t: t时刻样本，t: 采样时间步
    """
    B = x0.shape[0]
    
    z = torch.randn_like(x0, device=device)
    t = sample_time_steps(B, device)
    
    x_t = get_flow_matching_x_t(x0, z, t)
    x0_pred = model(x_t, t, cond)
    
    v_t_pred = get_flow_matching_target_velocity(x0_pred, x_t, t)
    v_t_target = get_flow_matching_target_velocity(x0, x_t, t)
    
    return v_t_pred, v_t_target, x_t, t


def generate_samples_ode(model, cond, shape, cfg_scale_interval, num_ode_steps=50, device=None):
    """
    通过ODE采样生成样本（Euler方法，简化版）
    Args:
        model: 模型实例
        cond: (B, C_cond, H, W)，条件输入
        shape: 生成样本的形状 (B, C, H, W)
        num_ode_steps: ODE采样步数
        device: 计算设备
    Returns:
        gen_sample: (B, C, H, W)，生成的样本
    """
    B, C, H, W = shape
    x_t = torch.randn(shape, device=device)  # 初始噪声z
    dt = 1.0 / num_ode_steps
    
    for step in range(num_ode_steps):
        t = torch.ones(B, device=device, requires_grad=False) * (step * dt)               
        x0_pred_cond = model(x_t, t, cond)
        v_t_pred_cond = get_flow_matching_target_velocity(x0_pred_cond, x_t, t)
        x0_pred_uncond = model(x_t, t, torch.zeros_like(cond, device=device, requires_grad=False))
        v_t_pred_uncond = get_flow_matching_target_velocity(x0_pred_uncond, x_t, t)
        v_t_pred = v_t_pred_uncond + cfg_scale_interval * (v_t_pred_cond - v_t_pred_uncond)
        x_t = x_t + v_t_pred * dt
    
    return x_t


def validate_model(model, val_loader, batch_transform, num_ode_steps, device, lpips_model, cfg_scale_interval):
    """
    改造验证逻辑：适配Flow Matching生成式模型
    先通过ODE采样生成样本，再计算评估指标
    """
    model.eval()
    val_losses, val_lpips, val_psnr = [], [], []
    gen_sample, val_gt, val_lq = None, None, None
    
    with torch.no_grad():
        pbar = tqdm(
            iterable=enumerate(val_loader),
            desc="Validating",
            total=len(val_loader),
            unit="batch",
            leave=False
        )
        for i, val_batch in pbar:
            to(val_batch, device)
            val_batch = batch_transform(val_batch)
            val_lq, val_gt, val_dpm = val_batch
            
            val_cond = torch.cat([val_lq, val_dpm], dim=1)
            gen_sample = generate_samples_ode(
                model=model,
                cond=val_cond,
                shape=val_gt.shape,
                cfg_scale_interval=cfg_scale_interval,
                num_ode_steps=num_ode_steps,
                device=device
            )
            
            val_losses.append(F.mse_loss(gen_sample, val_gt, reduction="mean"))
            val_lpips.append(lpips_model(gen_sample, val_gt, normalize=True).mean().item())
            val_psnr.append(calculate_psnr_pt(gen_sample, val_gt, crop_border=0).mean().item())
            
            if val_losses:
                avg_loss = sum(val_losses) / len(val_losses)
                pbar.set_postfix({"Avg Loss": f"{avg_loss:.6f}"})
    
    model.train()

    return {
        "val_losses": val_losses,
        "val_lpips": val_lpips,
        "val_psnr": val_psnr,
        "gen_sample": gen_sample,
        "val_gt": val_gt,
        "val_lq": val_lq,
    }
