from itertools import cycle
import os
from argparse import ArgumentParser
import warnings
from omegaconf import OmegaConf
import torch
from utils.model.gan import Discriminator
from utils.model.palette import UNet as Generator
# from utils.model.gan import Generator
# from utils.model.swinir import SwinIR as Generator
from torch.utils.data import DataLoader
from utils.dataset.dataset import MicroscopyDeepZDataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import lpips
from torch.utils.tensorboard import SummaryWriter
from utils.common import instantiate_from_config, to
from utils.log.utils import normalize_images, log_images
from utils.gan.loss import GAN_Loss
from utils.gan.val import validate_model


def train(args) -> None:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    
    torch.set_float32_matmul_precision('medium')
    
    best_val_loss = float('inf')
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    if accelerator.is_main_process:
        exp_dir = cfg.train.gan.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    generator: Generator = instantiate_from_config(cfg.model.gan.generator)
    discriminator: Discriminator = instantiate_from_config(cfg.model.gan.discriminator)

    total_gen_params = sum(p.numel() for p in generator.parameters())
    total_dis_params = sum(p.numel() for p in discriminator.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in generator.parameters()) / (1024 * 1024)
    
    if accelerator.is_local_main_process:
        print(f"Number of generator parameters: {total_gen_params:,}")
        print(f"Number of discriminator parameters: {total_dis_params:,}")        
        print(f"Model size: {model_size_mb:.2f} MB")

    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.train.gan.learning_rate_g, weight_decay=0.01)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg.train.gan.learning_rate_d, weight_decay=0.01)
        
    if cfg.dataset.ch >= 0:
        dataset = MicroscopyDeepZDataset(pairs_file_path=f'data/pairs/train/ch{cfg.dataset.ch}_train_pairs.txt', image_size=(256, 256))
        val_dataset = MicroscopyDeepZDataset(pairs_file_path=f'data/pairs/val/ch{cfg.dataset.ch}_val_pairs.txt', image_size=(256, 256)) 
    else:
        raise ValueError

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        drop_last=False,
    )

    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    generator.train().to(device)
    discriminator.train().to(device)
    generator, opt_g, loader, val_loader = accelerator.prepare(generator, opt_g, loader, val_loader)
    discriminator, opt_d, loader, val_loader = accelerator.prepare(discriminator, opt_d, loader, val_loader)
    pure_generator = accelerator.unwrap_model(generator)

    gan_loss = GAN_Loss(
        adv_weight=cfg.train.gan.adv_weight, 
        content_alpha=cfg.train.gan.content_alpha,
        # gp_lambda=cfg.train.gan.gp_lambda
    )

    global_step = 0
    max_steps = cfg.train.train_steps
    warm_up_steps_d = cfg.train.gan.warm_up_steps_d
    step_loss_g, step_loss_d = [], []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpips_model = (
            lpips.LPIPS(net="alex", verbose=accelerator.is_local_main_process)
            .eval()
            .to(device)
        )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training Model with Flow Matching for {max_steps} steps...")
    
    G_STEP = cfg.train.gan.g_step
    D_STEP = cfg.train.gan.d_step
    loader_iter = iter(cycle(loader))
    
    pbar = tqdm(
        total=max_steps + warm_up_steps_d,
        disable=not accelerator.is_main_process,
        unit="step",
        desc="GAN Training"
    )
    
    while global_step < max_steps + warm_up_steps_d:
        batch = next(loader_iter)
        to(batch, device)
        batch = batch_transform(batch)
        lq, gt, dpm = batch

        if global_step >= warm_up_steps_d and global_step % (G_STEP + D_STEP) <= (G_STEP - 1):
            gen_img = generator(lq, dpm)
            fake_pred_g = discriminator(gen_img)
            
            g_loss, g_adv_loss, g_content_loss = gan_loss.compute_generator_loss(
                gen_img=gen_img,
                real_img=gt,
                fake_pred=fake_pred_g
            )
            
            with torch.no_grad():
                d_loss, d_loss_real, d_loss_fake = gan_loss.compute_discriminator_loss(
                    real_pred=discriminator(gt),
                    fake_pred=fake_pred_g
                )

            opt_g.zero_grad()
            accelerator.backward(g_loss) 
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            opt_g.step()
            accelerator.wait_for_everyone()
            step_loss_g.append(g_loss.item())

        else:            
            with torch.no_grad():
                gen_img = generator(lq, dpm)
                fake_pred_g = discriminator(gen_img)
                g_loss, g_adv_loss, g_content_loss = gan_loss.compute_generator_loss(
                    gen_img=gen_img,
                    real_img=gt,
                    fake_pred=fake_pred_g
                )

            real_pred = discriminator(gt)
            fake_pred = discriminator(gen_img.detach())                

            d_loss, d_loss_real, d_loss_fake = gan_loss.compute_discriminator_loss(
                # critic=discriminator,
                # real_data=gt,
                # fake_data=gen_img,
                real_pred=real_pred,
                fake_pred=fake_pred
            )

            opt_d.zero_grad()
            accelerator.backward(d_loss)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            opt_d.step()
            accelerator.wait_for_everyone()
            step_loss_d.append(d_loss.item())
        
        global_step += 1

        if global_step % cfg.train.log_every == 0 and global_step > (warm_up_steps_d + G_STEP + D_STEP - 1):
            avg_loss_g = (accelerator.gather(torch.tensor(step_loss_g, device=device).unsqueeze(0)).mean().item())
            avg_loss_d = (accelerator.gather(torch.tensor(step_loss_d, device=device).unsqueeze(0)).mean().item())
            step_loss_g.clear()
            step_loss_d.clear()
            if accelerator.is_main_process:
                writer.add_scalar("loss/gen_step", avg_loss_g, global_step)
                writer.add_scalar("loss/gen_adv_step", g_adv_loss.item(), global_step)
                writer.add_scalar("loss/gen_content_step", g_content_loss.item(), global_step)
                writer.add_scalar("loss/dis_step", avg_loss_d, global_step)
                writer.add_scalar("loss/dis_real_step", d_loss_real.item(), global_step)
                writer.add_scalar("loss/dis_fake_step", d_loss_fake.item(), global_step)
        
        if global_step % cfg.train.val_every == 0 and global_step > (warm_up_steps_d + G_STEP + D_STEP - 1):
            val_results = validate_model(
                generator=generator,
                discriminator=discriminator,
                val_loader=val_loader,
                batch_transform=batch_transform,
                device=device,
                lpips_model=lpips_model, 
                adv_weight=cfg.train.gan.adv_weight,
                content_alpha=cfg.train.gan.content_alpha,
            )
            val_losses, val_lpips, val_psnr = val_results["val_losses"], val_results["val_lpips"], val_results["val_psnr"]
            avg_val_loss = (accelerator.gather(torch.tensor(val_losses, device=device).unsqueeze(0)).mean().item())
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
                        checkpoint = pure_generator.state_dict()
                        ckpt_path = f"{ckpt_dir}/best_val_loss.pt"
                        torch.save(checkpoint, ckpt_path)
                        print(f"New best validation loss: {best_val_loss:.6f} at step {global_step}, saved to {ckpt_path}")
                
                gen_sample, val_gt, val_lq = val_results["gen_sample"], val_results["val_gt"], val_results["val_lq"]
                N = min(4, val_gt.shape[0])
                gen_norm, gt_norm, lq_norm = normalize_images(gen_sample[:N], val_gt[:N], val_lq[:N])
                if accelerator.is_main_process:
                    log_images(
                        writer, 
                        gen_sample[:N], gen_norm, 
                        val_gt[:N], gt_norm, 
                        val_lq[:N], lq_norm, 
                        global_step
                    )
        accelerator.wait_for_everyone()
        
        # 更新pbar进度与描述（包含D损失）
        pbar.update(1)
        pbar.set_postfix({
            "Gen Loss": f"{g_loss.item():.6f}",
            "Dis Loss": f"{d_loss.item():.6f}",
            "Global Step": f"{global_step:07d}"
        })
    
    pbar.close()

    if accelerator.is_main_process:
        print("GAN training done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml", type=str)
    args = parser.parse_args()
    train(args)