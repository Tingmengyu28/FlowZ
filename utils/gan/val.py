import torch
from tqdm import tqdm
from utils.common import to, calculate_psnr_pt
import torch.nn.functional as F


def validate_model(generator, discriminator, val_loader, batch_transform, device, lpips_model, adv_weight, content_alpha):
    """
    改造验证逻辑：适配Flow Matching生成式模型
    先通过ODE采样生成样本，再计算评估指标
    修正：扩展ODE采样中时间步t的维度，解决广播错误
    """
    generator.eval()
    discriminator.eval()
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
            
            gen_sample = generator(val_lq, val_dpm)

            val_loss = F.l1_loss(gen_sample, val_gt, reduction="mean")

            
            val_losses.append(val_loss.item())
            val_lpips.append(lpips_model(gen_sample, val_gt, normalize=True).mean().item())
            val_psnr.append(calculate_psnr_pt(gen_sample, val_gt, crop_border=0).mean().item())
            
            if val_losses:      
                avg_loss = sum(val_losses) / len(val_losses)
                pbar.set_postfix({"Avg Loss": f"{avg_loss:.6f}"})
            
    generator.train()
    discriminator.train()

    return {
        "val_losses": val_losses,
        "val_lpips": val_lpips,
        "val_psnr": val_psnr,
        "gen_sample": gen_sample,
        "val_gt": val_gt,
        "val_lq": val_lq,
    }
