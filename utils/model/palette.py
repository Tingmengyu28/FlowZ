import math
import torch
from torch import nn
from inspect import isfunction
from torch.utils.checkpoint import checkpoint

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=[1, 2, 4, 8, 8],
        attn_res=[8],
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        use_gradient_checkpointing=True,
        attn_n_head=1,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attn_n_head = attn_n_head

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, 
                    noise_level_emb_dim=noise_level_channel, 
                    norm_groups=norm_groups, 
                    dropout=dropout, 
                    with_attn=use_attn,
                    attn_n_head=self.attn_n_head  # 传递注意力头数
                ))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, 
                               noise_level_emb_dim=noise_level_channel, 
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True,
                               attn_n_head=self.attn_n_head),
            ResnetBlocWithAttn(pre_channel, pre_channel, 
                               noise_level_emb_dim=noise_level_channel, 
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False,
                               attn_n_head=self.attn_n_head)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, 
                    noise_level_emb_dim=noise_level_channel, 
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn,
                    attn_n_head=self.attn_n_head))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time, c):
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None
        x = torch.cat([c, x], dim=1)

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                if self.use_gradient_checkpointing and self.training:
                    x = checkpoint(layer, x, t, use_reentrant=False)
                else:
                    x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                if self.use_gradient_checkpointing and self.training:
                    x = checkpoint(layer, x, t, use_reentrant=False)
                else:
                    x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                if self.use_gradient_checkpointing and self.training:
                    x = checkpoint(layer, torch.cat((x, feats.pop()), dim=1), t, use_reentrant=False)
                else:
                    x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


# PositionalEncoding 保持不变（无显存优化点）
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


# 优化FeatureWiseAffine：减少张量view的冗余操作
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        out_dim = out_channels * (2 if use_affine_level else 1)
        self.noise_func = nn.Linear(in_channels, out_dim)

    def forward(self, x, noise_embed):
        # 预计算shape，减少重复操作
        b, c, h, w = x.shape
        noise_out = self.noise_func(noise_embed)
        if self.use_affine_level:
            gamma, beta = noise_out.chunk(2, dim=1)
            # 直接reshape，避免多次view
            gamma = gamma.view(b, -1, 1, 1)
            beta = beta.view(b, -1, 1, 1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise_out.view(b, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# 优化Upsample/Downsample：使用更高效的卷积实现
class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 比nearest更稳定，显存一致
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=False)  # 移除bias，减少参数和显存

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用步幅卷积，替代手动下采样，减少张量操作
        self.conv = nn.Conv2d(dim, dim, 4, 2, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


# Block 小幅优化：移除冗余的Identity
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        layers = [
            nn.GroupNorm(groups, dim),
            Swish(),
        ]
        if dropout > 0:  # 仅当dropout>0时添加，减少冗余
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Conv2d(dim, dim_out, 3, padding=1, bias=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ResnetBlock 无核心优化（已在FeatureWiseAffine中优化）
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1, bias=False) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()
        self.n_head = n_head
        self.head_dim = in_channel // n_head
        assert self.head_dim * n_head == in_channel, "in_channel must be divisible by n_head"
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.proj_out = nn.Conv2d(in_channel, in_channel, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).view(b, 3, self.n_head, self.head_dim, h*w)
        q, k, v = qkv.unbind(1)
        
        attn = (q.transpose(-1, -2) @ k) * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(b, c, h, w)
        
        return self.proj_out(out) + x


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False, attn_n_head=1):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, n_head=attn_n_head, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


# 工具函数保持不变
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d