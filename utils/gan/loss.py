import torch
import torch.nn as nn
import torch.autograd as autograd


class WGANGPAdversarialLoss(nn.Module):
    """
    WGAN-GP（梯度惩罚版）对抗损失类
    核心：Lipschitz连续性约束 + 无界Critic输出 + 保留计算图完整性
    """
    def __init__(self, gp_lambda=10.0):
        super().__init__()
        self.gp_lambda = gp_lambda  # 论文推荐默认10

    def _gradient_penalty(self, critic, real_data, fake_data, device):
        """
        计算梯度惩罚项（核心修正：retain_graph=True，保留计算图）
        """
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
        if len(real_data.shape) != 4:
            epsilon = epsilon.view(batch_size, *[1]*(len(real_data.shape)-1))
        
        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates.requires_grad_(True)

        critic_interpolates = critic(interpolates)

        gradients = autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.norm(gradients, p=2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        total_gp = self.gp_lambda * gradient_penalty

        del gradient_norm, gradient_penalty  # 这两个无计算图依赖，可安全删除
        return total_gp

    def critic_loss(self, critic, real_data, fake_data, real_pred, fake_pred):
        """
        计算Critic完整损失（保留计算图完整性）
        """
        device = real_data.device
        
        loss_real = -torch.mean(real_pred)
        loss_fake = torch.mean(fake_pred)
        base_loss = loss_real + loss_fake

        gp_loss = self._gradient_penalty(critic, real_data, fake_data, device)
        total_c_loss = base_loss + gp_loss

        return total_c_loss, loss_real, loss_fake, gp_loss
    
    def critic_base_loss(self, real_pred, fake_pred):
        """
        计算Critic基础损失（无梯度惩罚，用于生成器训练阶段）
        """
        loss_real = -torch.mean(real_pred)
        loss_fake = torch.mean(fake_pred)
        base_loss = loss_real + loss_fake
        return base_loss, loss_real, loss_fake

    def generator_loss(self, fake_pred):
        """
        计算WGAN-GP生成器损失
        """
        total_g_adv_loss = -torch.mean(fake_pred)
        return total_g_adv_loss


class LSGANAdversarialLoss(nn.Module):
    """
    对齐公式的LSGAN最小二乘对抗损失
    """
    def __init__(self):
        super().__init__()
        self.mse_criterion = nn.MSELoss()

    def discriminator_loss(self, real_pred, fake_pred):
        """
        判别器损失（完全匹配公式）
        :param real_pred: 判别器对真实样本z的输出 (B, 1)
        :param fake_pred: 判别器对生成样本G(x)的输出 (B, 1)
        :return: 判别器总损失
        """
        real_target = torch.ones_like(real_pred, device=real_pred.device)
        fake_target = torch.zeros_like(fake_pred, device=fake_pred.device)
        
        loss_real = self.mse_criterion(real_pred, real_target)  # 1/N * sum[(D(z)-1)^2]
        loss_fake = self.mse_criterion(fake_pred, fake_target)  # 1/N * sum[(D(G(x)))^2]
        
        total_d_loss = (loss_real + loss_fake) * 0.5
        return total_d_loss, loss_real, loss_fake

    def generator_loss(self, fake_pred):
        """
        生成器对抗损失（修正后匹配公式）
        :param fake_pred: 判别器对生成样本的输出 (B, 1)
        :return: 生成器对抗损失（带0.5权重）
        """
        fake_target = torch.ones_like(fake_pred, device=fake_pred.device)
        total_g_adv_loss = 0.5 * self.mse_criterion(fake_pred, fake_target)
        return total_g_adv_loss


class MAEContentLoss(nn.Module):
    """
    对齐公式的MAE内容损失（保留原有逻辑）
    """
    def __init__(self, alpha=100.0):
        super().__init__()
        self.mae_criterion = nn.L1Loss()
        self.alpha = alpha

    def __call__(self, gen_img, real_img):
        """
        计算加权MAE损失
        """
        mae_loss = self.mae_criterion(gen_img, real_img)
        return self.alpha * 0.5 * mae_loss

# class GAN_Loss(nn.Module):
#     """
#     WGAN-GP版Deep-Z cGAN总损失（修正：避免重复构建计算图）
#     """
#     def __init__(self, adv_weight=1.0, content_alpha=100.0, gp_lambda=10.0):
#         super().__init__()
#         self.adv_loss = WGANGPAdversarialLoss(gp_lambda=gp_lambda)
#         self.content_loss = MAEContentLoss(alpha=content_alpha)
#         self.adv_weight = adv_weight

#     def compute_discriminator_loss(self, critic, real_data, fake_data, real_pred, fake_pred):
#         """
#         计算Critic总损失
#         """
#         total_c_loss, loss_real, loss_fake, gp_loss = self.adv_loss.critic_loss(
#             critic, real_data, fake_data, real_pred, fake_pred
#         )
#         return total_c_loss, loss_real, loss_fake, gp_loss

#     def compute_generator_loss(self, gen_img, real_img, fake_pred):
#         """
#         修正：返回已计算的损失变量，不重复构建计算图
#         """
#         g_adv_loss = self.adv_weight * self.adv_loss.generator_loss(fake_pred)
#         g_content_loss = self.content_loss(gen_img, real_img)
#         total_g_loss = g_adv_loss + g_content_loss

#         return total_g_loss, g_adv_loss, g_content_loss


class GAN_Loss(nn.Module):
    """
    完全对齐公式的Deep-Z cGAN总损失
    """
    def __init__(self, adv_weight=1.0, content_alpha=100.0):
        super().__init__()
        self.adv_loss = LSGANAdversarialLoss()
        self.content_loss = MAEContentLoss(alpha=content_alpha)
        self.adv_weight = adv_weight

    def compute_discriminator_loss(self, real_pred, fake_pred):
        """计算判别器总损失（直接调用对齐后的LSGAN损失）"""
        total_d_loss, loss_real, loss_fake = self.adv_loss.discriminator_loss(real_pred, fake_pred)
        return total_d_loss, loss_real, loss_fake

    def compute_generator_loss(self, gen_img, real_img, fake_pred):
        """计算生成器总损失（对抗损失+内容损失，完全匹配公式）"""
        g_adv_loss = self.adv_weight * self.adv_loss.generator_loss(fake_pred)
        g_content_loss = self.content_loss(gen_img, real_img)
        total_g_loss = g_adv_loss + g_content_loss
        return total_g_loss, g_adv_loss, g_content_loss