import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """时间步嵌入层：将1-D batch timesteps转换为高维嵌入向量，融入UNet特征"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, timesteps):
        t_emb = timesteps.unsqueeze(-1).float()
        t_emb = self.mlp(t_emb)
        t_emb = t_emb.reshape(t_emb.shape[0], self.embed_dim, 1, 1)  # 直接重塑为目标形状
        return t_emb

class ConditionEmbedder(nn.Module):
    """条件嵌入层：处理cond（N x 2 x H x W），转换为适配UNet输入的特征图"""
    def __init__(self, cond_in_channels=2, embed_out_channels=25):
        super().__init__()
        self.conv = nn.Conv2d(
            cond_in_channels,
            embed_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='replicate'
        )
        self.relu = nn.ReLU()
    
    def forward(self, cond):
        return self.relu(self.conv(cond))

class DownBlock(nn.Module):
    """下采样块（参考Deep-Z公式(1)）：残差连接+零填充通道匹配"""
    def __init__(self, in_channels, k1, k2):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        
        self.conv_branch = nn.Sequential(
            nn.Conv2d(
                in_channels, k1, kernel_size=3, stride=1, padding=1, padding_mode='replicate'
            ),
            nn.ReLU(),
            nn.Conv2d(
                k1, k2, kernel_size=3, stride=1, padding=1, padding_mode='replicate'
            ),
            nn.ReLU()
        )
        
        self.in_channels = in_channels
        self.out_channels = k2
    
    def _zero_pad_channels(self, x):
        """对输入x进行通道维度零填充，匹配输出通道数k2"""
        pad_channels = self.out_channels - self.in_channels
        if pad_channels > 0:
            return F.pad(x, (0, 0, 0, 0, 0, pad_channels))
        return x
    
    def forward(self, x, t_emb):
        if t_emb.shape[1] >= self.in_channels:
            x = x + t_emb[:, :self.in_channels, :, :]
        
        conv_out = self.conv_branch(x)
        x_padded = self._zero_pad_channels(x)
        out = x_padded + conv_out
        
        return out

class UpBlock(nn.Module):
    """上采样块（参考Deep-Z公式(2)）：通道拼接+卷积处理"""
    def __init__(self, in_channels, k3, k4):
        super().__init__()
        self.k3 = k3
        self.k4 = k4
        
        self.conv_branch = nn.Sequential(
            nn.Conv2d(
                in_channels, k3, kernel_size=3, stride=1, padding=1, padding_mode='replicate'
            ),
            nn.ReLU(),
            nn.Conv2d(
                k3, k4, kernel_size=3, stride=1, padding=1, padding_mode='replicate'
            ),
            nn.ReLU()
        )
        
        self.out_channels = k4
    
    def forward(self, x, skip_x, t_emb):
        x_concat = torch.cat([x, skip_x], dim=1)
        
        if t_emb.shape[1] >= x_concat.shape[1]:
            x_concat = x_concat + t_emb[:, :x_concat.shape[1], :, :]
        
        out = self.conv_branch(x_concat)
        return out

class UNetModel(nn.Module):
    """
    用于Diffusion Model的UNet网络（纠错版）：
    修复上采样路径转置卷积通道数配置错误，确保维度流转一致
    严格支持x（N x C x ...）、cond（N x 2 x ...）输入，输出x_{t-1}（N x C x ...）
    """
    def __init__(
        self,
        image_size: int,
        in_channel: int,
        out_channel: int,
        device: torch.device = None
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.down_k1 = [25, 72, 144, 288, 576]
        self.down_k2 = [48, 96, 192, 384, 768]
        self.up_k3 = [576, 288, 144, 72]
        self.up_k4 = [384, 192, 96, 48]
        
        self.timestep_embed_dim = self.down_k2[-1]  # 768，适配最后一个下采样块
        self.timestep_embedder = TimestepEmbedder(embed_dim=self.timestep_embed_dim)
        self.condition_embedder = ConditionEmbedder(
            cond_in_channels=2,
            embed_out_channels=self.down_k1[0]
        )
        
        self.input_proj = nn.Conv2d(
            in_channel,
            self.down_k1[0],
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='replicate'
        )
        
        self.down_blocks = nn.ModuleList()
        self.max_pools = nn.ModuleList()
        down_in_channels = [self.down_k1[0]] + self.down_k2[:-1]
        
        for i in range(5):  # 5个下采样块，通道流转正确
            in_ch = down_in_channels[i]
            k1 = self.down_k1[i]
            k2 = self.down_k2[i]
            self.down_blocks.append(DownBlock(in_ch, k1, k2))
            if i < 4:
                self.max_pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        up_conv_in_channels = [self.down_k2[4], self.up_k4[0], self.up_k4[1], self.up_k4[2]]  # [768, 384, 192, 96]
        up_conv_out_channels = [self.up_k4[0], self.up_k4[1], self.up_k4[2], self.up_k4[3]]  # [384, 192, 96, 48]
        
        up_block_in_channels = [
            self.up_k4[0] + self.down_k2[3],  # 384 + 384 = 768
            self.up_k4[1] + self.down_k2[2],  # 192 + 192 = 384
            self.up_k4[2] + self.down_k2[1],  # 96 + 96 = 192
            self.up_k4[3] + self.down_k2[0]   # 48 + 48 = 96
        ]
        
        for i in range(4):  # 4个上采样块，通道数严格匹配
            self.up_convs.append(nn.ConvTranspose2d(
                in_channels=up_conv_in_channels[i],
                out_channels=up_conv_out_channels[i],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0  # 避免空间维度偏移
            ))
            in_ch = up_block_in_channels[i]
            k3 = self.up_k3[i]
            k4 = self.up_k4[i]
            self.up_blocks.append(UpBlock(in_ch, k3, k4))
        
        self.final_conv = nn.Conv2d(
            self.up_k4[3],
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='replicate'
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重Xavier初始化，偏置初始化为0.1"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        """
        前向传播（纠错后：通道流转无偏差）
        Args:
            x: 带噪图像 (N, C, H, W) → 严格支持N x C x ...
            timesteps: 时间步 (N,) → 1-D batch
            cond: 条件输入 (N, 2, H, W) → 严格支持N x 2 x ...
        Returns:
            x_t_minus_1: 预测的x_{t-1} (N, C, H, W) → 与输入x维度完全一致
        """
        assert x.dim() == 4, f"输入x必须为4维(N, C, H, W)，当前为{x.dim()}维"
        assert cond.dim() == 4 and cond.shape[1] == 2, f"条件cond必须为4维(N, 2, H, W)，当前为{cond.shape}"
        assert x.shape[2] == x.shape[3] == self.image_size, \
            f"输入图像尺寸必须为{self.image_size}x{self.image_size}，当前为{x.shape[2]}x{x.shape[3]}"
        assert cond.shape[2:] == x.shape[2:], \
            f"条件与图像空间维度不匹配，图像{x.shape[2:]}，条件{cond.shape[2:]}"
        
        t_emb = self.timestep_embedder(timesteps)  # (N, 768, 1, 1)
        cond_emb = self.condition_embedder(cond)    # (N, 25, H, W) → 处理N x 2 x ...输入
        
        x_proj = self.input_proj(x)  # (N, C) → (N, 25)，适配下采样输入
        x_feat = x_proj + cond_emb   # 条件引导融入，通道数25，无维度偏差
        
        skip_features = []
        for i, down_block in enumerate(self.down_blocks):
            x_feat = down_block(x_feat, t_emb)
            if i < 4:
                skip_features.append(x_feat)
                x_feat = self.max_pools[i](x_feat)
        
        skip_features = skip_features[::-1]
        
        for i, (up_conv, up_block) in enumerate(zip(self.up_convs, self.up_blocks)):
            x_feat = up_conv(x_feat)
            x_feat = up_block(x_feat, skip_features[i], t_emb)
        
        x_t_minus_1 = self.final_conv(x_feat)
        
        assert x_t_minus_1.shape == x.shape, \
            f"输出维度与输入不匹配，输入{x.shape}，输出{x_t_minus_1.shape}"
        
        return x_t_minus_1