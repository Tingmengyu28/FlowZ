import os
from itertools import cycle
from argparse import ArgumentParser
import warnings
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataset.dataset import MicroscopyDeepZDataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import lpips
from torch.utils.tensorboard import SummaryWriter
from utils.common import instantiate_from_config, to
from utils.log.utils import normalize_images, log_images
from utils.flow.v_pred import flow_matching_forward_pass, validate_model, drop_labels
from utils.ema import EMA


def train(args) -> None:
    cfg = OmegaConf.load(args.config)
    
    accelerator = Accelerator(
        split_batches=True,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps
    )
    num_processes = accelerator.num_processes
    set_seed(231, device_specific=True)
    
    best_val_loss = float('inf')
    device = accelerator.device
    accelerator.free_memory()
    
    if accelerator.is_main_process:
        exp_dir = cfg.train.fm.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    unet = instantiate_from_config(cfg.model.palette)
    
    ema_decay = getattr(cfg.train.fm, "ema_decay", cfg.train.ema_decay)
    ema = EMA(unet, decay=ema_decay)
    ema = accelerator.prepare(ema)
    
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in unet.parameters()) / (1024 * 1024)
    
    if accelerator.is_local_main_process:
        print(f"Number of parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size_mb:.2f} MB")
        print(f"EMA initialized with decay: {ema.decay}")

    # Setup optimizer:
    opt = torch.optim.AdamW(
        unet.parameters(), 
        lr=cfg.train.fm.learning_rate,
        betas=(0.5, 0.999),
        weight_decay=0.01,
        foreach=False,
    )
        
    if cfg.dataset.ch >= 0:
        if cfg.dataset.name == 'exp':
            dataset = MicroscopyDeepZDataset(pairs_file_path=f'data/pairs/ch{cfg.dataset.ch}/train_pairs.txt', image_size=(256, 256))
            val_dataset = MicroscopyDeepZDataset(pairs_file_path=f'data/pairs/ch{cfg.dataset.ch}/val_pairs.txt', image_size=(256, 256))
        else:
            dataset = MicroscopyDeepZDataset(pairs_file_path=f'data_{cfg.dataset.name}/pairs/train_pairs.txt', image_size=(256, 256))
            val_dataset = MicroscopyDeepZDataset(pairs_file_path=f'data_{cfg.dataset.name}/pairs/val_pairs.txt', image_size=(256, 256))
    else:
        raise ValueError

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    unet.train().to(device)
    unet, opt, loader, val_loader = accelerator.prepare(
        unet, opt, loader, val_loader
    )
    pure_unet = accelerator.unwrap_model(unet)

    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpips_model = (
            lpips.LPIPS(net="alex", verbose=accelerator.is_local_main_process)
            .eval()
            .to(device)
        )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training Model with Flow Matching + EMA for {max_steps} steps...")
    
    loader_iter = iter(cycle(loader))
    pbar = tqdm(
        desc="Flow Matching (EMA)",
        disable=not accelerator.is_main_process,
        unit="step",
        total=max_steps,
    )
    
    while global_step < max_steps:
        if global_step % cfg.train.val_every == 0 and global_step > 0:
            if accelerator.is_main_process:
                torch.cuda.empty_cache()
        
        batch = next(loader_iter)
        to(batch, device)
        batch = batch_transform(batch)
        lq, gt, dpm = batch
        
        cond = torch.cat([lq, dpm], dim=1)
        cond_dropped = drop_labels(cond, label_drop_prob=cfg.train.fm.label_drop_prob)
        
        v_t_pred, v_t_target, x_t, t = flow_matching_forward_pass(
            model=unet,
            x0=gt,
            cond=cond_dropped,
            alpha=cfg.train.fm.Beta_alpha,
            beta=cfg.train.fm.Beta_beta,
            device=device,
        )
            
        loss = F.mse_loss(v_t_pred, v_t_target, reduction="mean")
        
        opt.zero_grad()
        accelerator.backward(loss)
        opt.step()
        accelerator.wait_for_everyone()

        ema.step()

        step_loss.append(loss.item())
        global_step += 1        

        # Log loss values:
        if global_step % cfg.train.log_every == 0 and global_step > 0:
            avg_loss = (accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item())
            step_loss.clear()
            if accelerator.is_main_process:
                writer.add_scalar("loss/flow_matching_step", avg_loss, global_step)

        # Evaluate model:
        if global_step % cfg.train.val_every == 0 and global_step > 0:
            ema.store()
            ema.copy_to()

            val_results = validate_model(
                unet, val_loader, 
                batch_transform, 
                num_ode_steps=cfg.inference.fm.num_ode_steps,
                device=device, 
                lpips_model=lpips_model,
                cfg_scale_interval=cfg.train.fm.cfg_scale_interval,
                top_n=cfg.inference.fm.top_n // num_processes,
            )
            val_loss, val_lpips, val_psnr = val_results["val_losses"], val_results["val_lpips"], val_results["val_psnr"]
            avg_val_loss = (accelerator.gather(torch.tensor(val_loss, device=device).unsqueeze(0)).mean().item())
            avg_val_lpips = (accelerator.gather(torch.tensor(val_lpips, device=device).unsqueeze(0)).mean().item())
            avg_val_psnr = (accelerator.gather(torch.tensor(val_psnr, device=device).unsqueeze(0)).mean().item())
            
            if accelerator.is_local_main_process:
                for tag, val in [
                    ("val/loss", avg_val_loss),
                    ("val/lpips", avg_val_lpips),
                    ("val/psnr", avg_val_psnr),
                ]:
                    writer.add_scalar(tag, val, global_step)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if accelerator.is_main_process:
                        checkpoint = {
                            "model": pure_unet.state_dict(),
                            "ema": ema.state_dict(),
                            "step": global_step,
                            "best_val_loss": best_val_loss
                        }
                        ckpt_path = f"{ckpt_dir}/best_val_loss_ema.pt"
                        torch.save(checkpoint, ckpt_path)
                        print(f"New best EMA validation loss: {best_val_loss:.6f} at step {global_step}, saved to {ckpt_path}")

                gen_sample, val_gt, val_lq = val_results["gen_sample"], val_results["val_gt"], val_results["val_lq"]
                N = min(8, val_gt.shape[0])
                gen_norm, gt_norm, lq_norm = normalize_images(gen_sample[:N], val_gt[:N], val_lq[:N])
                if accelerator.is_main_process:
                    log_images(
                        writer, 
                        gen_sample[:N], gen_norm, 
                        val_gt[:N], gt_norm, 
                        val_lq[:N], lq_norm, 
                        global_step
                    )

            ema.restore()
            accelerator.wait_for_everyone()
        
        pbar.update(1)
        pbar.set_postfix({
            "Global Step": f"{global_step:07d}",
            "Flow Loss": f"{loss.item():.6f}",
            "EMA Decay": f"{ema.decay}"
        })
            
    if accelerator.is_main_process:
        ema.store()
        ema.copy_to()
        final_ckpt = {
            "model": pure_unet.state_dict(),
            "ema": ema.state_dict(),
            "step": global_step,
            "best_val_loss": best_val_loss
        }
        torch.save(final_ckpt, f"{ckpt_dir}/final_ema_model.pt")
        ema.restore()

    pbar.close()

    if accelerator.is_main_process:
        print("Flow Matching + EMA training done!")
        writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/params.yaml", type=str)
    args = parser.parse_args()
    train(args)
