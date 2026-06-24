import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.common import to, calculate_psnr_pt
from torch.distributions import Beta
from skimage.metrics import structural_similarity as ssim


def sample_time_steps(batch_size, alpha, beta, device):
    beta_dist = Beta(torch.tensor([alpha]), torch.tensor([beta]))
    t = beta_dist.sample((batch_size,)).squeeze(1).to(device)
    t = torch.clamp(t, min=1e-6, max=1.0 - 1e-6)
    
    return t


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
    t_expand = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_expand) * z + t_expand * x0
    return x_t


def get_flow_matching_target_velocity(x0, z):
    """
    计算Flow Matching的目标速度场v_t^target（CFM简化版）
    Args:
        x0: (B, C, H, W)，目标样本（gt）
        z: (B, C, H, W)，初始分布样本（噪声）
    Returns:
        v_t_target: (B, C, H, W)，目标速度场
    """
    return x0 - z


def flow_matching_forward_pass(model, x0, cond, alpha, beta, device):
    """
    Flow Matching单次前向传播（完整训练步骤）
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
    t = sample_time_steps(B, alpha=alpha, beta=beta, device=device)
    x_t = get_flow_matching_x_t(x0, z, t)
    v_t_pred = model(x_t, t, cond)
    v_t_target = get_flow_matching_target_velocity(x0, z)
    
    return v_t_pred, v_t_target, x_t, t


def validate_model(model, val_loader, batch_transform, num_ode_steps, device, lpips_model, cfg_scale_interval, top_n=4):
    """
    改造验证逻辑：适配Flow Matching生成式模型，仅推理前N张图片（同步优化pbar）
    先通过ODE采样生成样本，再计算评估指标
    """
    model.eval()
    val_losses, val_lpips, val_psnr, val_ssim = [], [], [], []
    gen_sample, val_gt, val_lq, val_dpm = None, None, None, None
    processed_imgs = 0
    
    with torch.no_grad():
        pbar = tqdm(
            desc="Validating (Top N imgs)",
            total=top_n,
            unit="img",
            leave=False,
            dynamic_ncols=True
        )
        
        for batch_idx, val_batch in enumerate(val_loader):
            if processed_imgs >= top_n:
                break
            
            val_batch = batch_transform(val_batch)
            val_lq_batch, val_gt_batch, val_dpm_batch = val_batch
            val_lq_batch, val_gt_batch, val_dpm_batch = val_lq_batch.to(device), val_gt_batch.to(device), val_dpm_batch.to(device)
            
            B, C, H, W = val_gt_batch.shape
            batch_need = min(B, top_n - processed_imgs)
            
            val_lq_crop = val_lq_batch[:batch_need]
            val_gt_crop = val_gt_batch[:batch_need]
            val_dpm_crop = val_dpm_batch[:batch_need]
            
            val_cond_crop = torch.cat([val_lq_crop, val_dpm_crop], dim=1).to(device)
            x_t = torch.randn_like(val_gt_crop, device=device)
            dt = 1.0 / num_ode_steps
            
            for step in range(num_ode_steps):
                t = torch.ones(batch_need, device=device) * (step * dt)
                v_cond = model(x_t, t, val_cond_crop)
                uncond_cond = torch.zeros_like(val_cond_crop, device=device, requires_grad=False)
                v_uncond_current = model(x_t, t, uncond_cond)
                v_final = cfg_scale_interval * v_cond + (1.0 - cfg_scale_interval) * v_uncond_current
                x_t = x_t + v_final * dt
            
            gen_sample_crop = torch.clamp(x_t, min=-1.0, max=1.0)
            
            if batch_idx == 0:
                gen_sample = gen_sample_crop
                val_gt = val_gt_crop
                val_lq = val_lq_crop
                val_dpm = val_dpm_crop
            
            batch_loss = F.l1_loss(gen_sample_crop, val_gt_crop, reduction="mean").item()
            batch_lpips = lpips_model(gen_sample_crop, val_gt_crop, normalize=True).mean().item()
            batch_psnr = calculate_psnr_pt(gen_sample_crop, val_gt_crop, crop_border=0).mean().item()
            
            # 计算SSIM
            gen_sample_np = gen_sample_crop.cpu().numpy()
            val_gt_np = val_gt_crop.cpu().numpy()
            batch_ssim = ssim(
                gen_sample_np[0, 0],
                val_gt_np[0, 0],
                data_range=2.0,
                channel_axis=None
            )
            
            val_losses.append(batch_loss)
            val_lpips.append(batch_lpips)
            val_psnr.append(batch_psnr)
            val_ssim.append(batch_ssim)
            processed_imgs += batch_need
            
            pbar.update(batch_need)
            
            if val_losses:
                avg_loss = sum(val_losses) / len(val_losses)
                pbar.set_postfix({
                    "Avg Loss": f"{avg_loss:.6f}",
                    "Processed": f"{processed_imgs}/{top_n}",
                    "Batch Loss": f"{batch_loss:.6f}"
                }, refresh=True)
        
        pbar.close()
    
    model.train()

    return {
        "val_losses": val_losses,
        "val_lpips": val_lpips,
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
        "gen_sample": gen_sample,
        "val_gt": val_gt,
        "val_lq": val_lq,
        "val_dpm": val_dpm,
    }