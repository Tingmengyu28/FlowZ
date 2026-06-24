# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class DownBlock(nn.Module):
#     """
#     生成器下采样块，实现论文公式(1)
#     xk+1 = xk + ReLU(CONVk2(ReLU(CONVk1(xk))))
#     残差连接，通道不匹配时对xk零填充；3×3卷积、replicate padding=1、步长=1
#     """
#     def __init__(self, in_channels, k1, k2):
#         super().__init__()
#         self.k1 = k1
#         self.k2 = k2
#         # 双层卷积：CONVk1 -> ReLU -> CONVk2 -> ReLU
#         self.conv_branch = nn.Sequential(
#             nn.Conv2d(in_channels, k1, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(k1, k2, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
#             nn.ReLU(inplace=False)
#         )
#         self.in_channels = in_channels
#         self.out_channels = k2

#     def _zero_pad_channels(self, x):
#         """对输入xk零填充通道，匹配输出xk+1的通道数（残差连接通道对齐）"""
#         pad_channels = self.out_channels - self.in_channels
#         if pad_channels > 0:
#             # 维度顺序：(B, C, H, W)，仅在通道维度填充
#             return F.pad(x, (0, 0, 0, 0, 0, pad_channels), mode='constant', value=0.0)
#         return x

#     def forward(self, x):
#         conv_out = self.conv_branch(x)
#         x_padded = self._zero_pad_channels(x)  # 通道对齐
#         out = x_padded + conv_out  # 残差连接
#         return out

# class UpBlock(nn.Module):
#     """
#     生成器上采样块，实现论文公式(2)
#     yk = ReLU(CONVk4(ReLU(CONVk3(CAT(xk+1, yk+1)))))
#     通道拼接（下采样特征+上采样特征）；3×3卷积、replicate padding=1、步长=1
#     """
#     def __init__(self, in_channels, k3, k4):
#         super().__init__()
#         self.k3 = k3
#         self.k4 = k4
#         # 双层卷积：CONVk3 -> ReLU -> CONVk4 -> ReLU
#         self.conv_branch = nn.Sequential(
#             nn.Conv2d(in_channels, k3, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(k3, k4, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
#             nn.ReLU(inplace=False)
#         )
#         self.out_channels = k4

#     def forward(self, x, skip_x):
#         """
#         x: 上采样路径的输入特征 yk+1
#         skip_x: 下采样路径的对应特征 xk+1（通道拼接用）
#         """
#         x_concat = torch.cat([x, skip_x], dim=1)  # 通道维度拼接 CAT(·)
#         out = self.conv_branch(x_concat)
#         return out

# class Generator(nn.Module):
#     """
#     Deep-Z生成器：5个下采样块 + 4个上采样块（对称）
#     下采样：k1=[25,72,144,288,576], k2=[48,96,192,384,768]
#     上采样：k3=[72,144,288,576], k4=[48,96,192,384]
#     块间：下采样=2×2MaxPool(步长2)；上采样=2×转置卷积(步长2)
#     输出：48通道 -> 1通道（论文指定）
#     """
#     def __init__(self, in_channel=1, cond_channel=2, image_size=256):
#         super().__init__()
#         self.image_size = image_size
#         # 论文指定的通道数（严格对齐）
#         self.down_k1 = [25, 72, 144, 288, 576]
#         self.down_k2 = [48, 96, 192, 384, 768]
#         self.up_k3 = [72, 144, 288, 576]
#         self.up_k4 = [48, 96, 192, 384]

#         # 输入投影：将「图像+条件」拼接后的特征映射到第一个下采样块的输入通道25
#         self.input_proj = nn.Conv2d(
#             in_channels=in_channel + cond_channel,
#             out_channels=self.down_k1[0],
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             padding_mode='replicate'
#         )

#         self.down_blocks = nn.ModuleList()
#         self.max_pools = nn.ModuleList()
#         down_in_chs = [self.down_k1[0]] + self.down_k2[:-1]  # 下采样块输入通道
#         for i in range(5):
#             self.down_blocks.append(DownBlock(down_in_chs[i], self.down_k1[i], self.down_k2[i]))
#             if i < 4:
#                 self.max_pools.append(nn.MaxPool2d(kernel_size=2, stride=2))

#         self.up_convs = nn.ModuleList()
#         self.up_blocks = nn.ModuleList()
#         up_conv_in_chs = [self.down_k2[4], self.up_k4[0], self.up_k4[1], self.up_k4[2]]
#         up_conv_out_chs = [self.up_k4[0], self.up_k4[1], self.up_k4[2], self.up_k4[3]]
#         up_block_in_chs = [
#             self.up_k4[0] + self.down_k2[3],
#             self.up_k4[1] + self.down_k2[2],
#             self.up_k4[2] + self.down_k2[1],
#             self.up_k4[3] + self.down_k2[0]
#         ]
#         for i in range(4):
#             self.up_convs.append(nn.ConvTranspose2d(
#                 in_channels=up_conv_in_chs[i],
#                 out_channels=up_conv_out_chs[i],
#                 kernel_size=2,
#                 stride=2,
#                 padding=0,
#                 output_padding=0,
#                 padding_mode='zeros'
#             ))
#             self.up_blocks.append(UpBlock(up_block_in_chs[i], self.up_k3[i], self.up_k4[i]))

#         self.final_conv = nn.Conv2d(
#             in_channels=self.up_k4[3],
#             out_channels=1,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             padding_mode='replicate'
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         """Xavier初始化权重，偏置固定为0.1（严格对齐论文）"""
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.1)

#     def forward(self, x, cond):
#         """
#         cGAN生成器前向：输入图像+条件拼接作为输入
#         x: 输入图像 (B, 1, H, W) （如低质图/噪声）
#         cond: 条件张量 (B, 2, H, W) （用户场景的dpm+lq拼接）
#         return: 生成图像 (B, 1, H, W)
#         """
#         x_cond = torch.cat([x, cond], dim=1)
#         x_feat = self.input_proj(x_cond)

#         skip_feats = []
#         for i in range(5):
#             x_feat = self.down_blocks[i](x_feat)
#             if i < 4:
#                 skip_feats.append(x_feat)
#                 x_feat = self.max_pools[i](x_feat)
#         skip_feats = skip_feats[::-1]

#         for i in range(4):
#             x_feat = self.up_convs[i](x_feat)  # 转置卷积上采样
#             x_feat = self.up_blocks[i](x_feat, skip_feats[i])  # 通道拼接+卷积

#         # 最终输出
#         gen_img = self.final_conv(x_feat)
#         return gen_img

# class DiscConvBlock(nn.Module):
#     """
#     判别器卷积块，实现论文公式(3)
#     zi+1 = LReLU(CONVi2(LReLU(CONVi1(zi))))
#     LReLU(0.01)，第一个卷积步长1，第二个卷积步长2（下采样2×）
#     3×3卷积、replicate padding=1
#     """
#     def __init__(self, in_channels, i1, i2):
#         super().__init__()
#         self.i1 = i1
#         self.i2 = i2
#         self.conv_branch = nn.Sequential(
#             nn.Conv2d(in_channels, i1, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
#             nn.LeakyReLU(negative_slope=0.01, inplace=False),
#             nn.Conv2d(i1, i2, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),  # 步长2下采样
#             nn.LeakyReLU(negative_slope=0.01, inplace=False)
#         )
#         self.out_channels = i2

#     def forward(self, x):
#         out = self.conv_branch(x)
#         return out


# class Discriminator(nn.Module):
#     """
#     Deep-Z判别器：6个卷积块 + 平均池化 + 全连接层 + Sigmoid
#     卷积块通道：i1=[48,96,192,384,768,1536], i2=[96,192,384,768,1536,3072]
#     输出：判别分数(0,1)，0=假，1=真
#     """
#     def __init__(self, in_channel=1, cond_channel=2):
#         super().__init__()
#         self.i1 = [48, 96, 192, 384, 768, 1536]
#         self.i2 = [96, 192, 384, 768, 1536, 3072]
#         self.fc_hidden = 3072  # 论文指定的全连接隐藏层维度

#         self.input_proj = nn.Conv2d(
#             in_channels=in_channel + cond_channel,
#             out_channels=self.i1[0],
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             padding_mode='replicate'
#         )

#         self.disc_blocks = nn.ModuleList()
#         disc_in_chs = [self.i1[0]] + self.i2[:-1]
#         for i in range(6):
#             self.disc_blocks.append(DiscConvBlock(disc_in_chs[i], self.i1[i], self.i2[i]))

#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化，兼容任意输入尺寸

#         self.fc_layers = nn.Sequential(
#             nn.Linear(self.fc_hidden, self.fc_hidden),
#             nn.LeakyReLU(negative_slope=0.01, inplace=False),
#             nn.Linear(self.fc_hidden, 1),
#             nn.Sigmoid()  # 输出分数(0,1)
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         """Xavier初始化权重，偏置固定为0.1（严格对齐论文）"""
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.1)

#     def forward(self, x, cond):
#         """
#         cGAN判别器前向：输入图像+条件拼接，输出判别分数
#         x: 图像（真实/生成）(B, 1, H, W)
#         cond: 条件张量 (B, 2, H, W)
#         return: 判别分数 (B, 1)，值∈(0,1)
#         """
#         x_cond = torch.cat([x, cond], dim=1)
#         x_feat = self.input_proj(x_cond)

#         for block in self.disc_blocks:
#             x_feat = block(x_feat)

#         x_pool = self.avg_pool(x_feat)
#         x_flat = x_pool.view(x_pool.size(0), -1)  # (B, 3072)

#         score = self.fc_layers(x_flat)
#         return score


import torch
import torch.nn as nn

# ---------------------
# 生成器 Generator
# ---------------------
class Generator(nn.Module):
    def __init__(self, in_channel=2, out_channel=1):
        super(Generator, self).__init__()
        # ========== 编码器（下采样阶段） ==========
        # 阶段1: 256×256 → 256×256 → 128×128
        self.enc1_conv1 = nn.Conv2d(in_channel, 25, kernel_size=3, padding=1)
        self.enc1_conv2 = nn.Conv2d(25, 48, kernel_size=3, padding=1)
        self.enc1_pool = nn.MaxPool2d(2, 2)  # 128×128

        # 阶段2: 128×128 → 128×128 → 64×64
        self.enc2_conv1 = nn.Conv2d(48, 72, kernel_size=3, padding=1)
        self.enc2_conv2 = nn.Conv2d(72, 96, kernel_size=3, padding=1)
        self.enc2_pool = nn.MaxPool2d(2, 2)  # 64×64

        # 阶段3: 64×64 → 64×64 → 32×32
        self.enc3_conv1 = nn.Conv2d(96, 144, kernel_size=3, padding=1)
        self.enc3_conv2 = nn.Conv2d(144, 192, kernel_size=3, padding=1)
        self.enc3_pool = nn.MaxPool2d(2, 2)  # 32×32

        # 阶段4: 32×32 → 32×32 → 16×16
        self.enc4_conv1 = nn.Conv2d(192, 288, kernel_size=3, padding=1)
        self.enc4_conv2 = nn.Conv2d(288, 384, kernel_size=3, padding=1)
        self.enc4_pool = nn.MaxPool2d(2, 2)  # 16×16

        # 阶段5: 瓶颈层 16×16 → 16×16
        self.enc5_conv1 = nn.Conv2d(384, 576, kernel_size=3, padding=1)
        self.enc5_conv2 = nn.Conv2d(576, 768, kernel_size=3, padding=1)

        # ========== 解码器（上采样阶段） ==========
        # 阶段1: 16×16 → 32×32
        self.dec1_up = nn.ConvTranspose2d(768, 576, kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv2d(576 + 384, 384, kernel_size=3, padding=1)  # 拼接编码器阶段4的输出
        self.dec1_conv2 = nn.Conv2d(384, 384, kernel_size=3, padding=1)

        # 阶段2: 32×32 → 64×64
        self.dec2_up = nn.ConvTranspose2d(384, 288, kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv2d(288 + 192, 288, kernel_size=3, padding=1)  # 拼接编码器阶段3的输出
        self.dec2_conv2 = nn.Conv2d(288, 288, kernel_size=3, padding=1)

        # 阶段3: 64×64 → 128×128
        self.dec3_up = nn.ConvTranspose2d(288, 144, kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv2d(144 + 96, 144, kernel_size=3, padding=1)  # 拼接编码器阶段2的输出
        self.dec3_conv2 = nn.Conv2d(144, 144, kernel_size=3, padding=1)

        # 阶段4: 128×128 → 256×256
        self.dec4_up = nn.ConvTranspose2d(144, 96, kernel_size=2, stride=2)
        self.dec4_conv1 = nn.Conv2d(96 + 48, 48, kernel_size=3, padding=1)  # 拼接编码器阶段1的输出
        self.dec4_conv2 = nn.Conv2d(48, out_channel, kernel_size=3, padding=1)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, vpm):
        # 输入拼接: image + VPM
        x = torch.cat([x, vpm], dim=1)  # 通道数: 1+1=2

        # ---------- 编码器前向 ----------
        # 阶段1
        e1 = self.relu(self.enc1_conv1(x))
        e1 = self.relu(self.enc1_conv2(e1))
        e1_pool = self.enc1_pool(e1)  # 128×128, 48

        # 阶段2
        e2 = self.relu(self.enc2_conv1(e1_pool))
        e2 = self.relu(self.enc2_conv2(e2))
        e2_pool = self.enc2_pool(e2)  # 64×64, 96

        # 阶段3
        e3 = self.relu(self.enc3_conv1(e2_pool))
        e3 = self.relu(self.enc3_conv2(e3))
        e3_pool = self.enc3_pool(e3)  # 32×32, 192

        # 阶段4
        e4 = self.relu(self.enc4_conv1(e3_pool))
        e4 = self.relu(self.enc4_conv2(e4))
        e4_pool = self.enc4_pool(e4)  # 16×16, 384

        # 阶段5（瓶颈）
        e5 = self.relu(self.enc5_conv1(e4_pool))
        e5 = self.relu(self.enc5_conv2(e5))  # 16×16, 768

        # ---------- 解码器前向 ----------
        # 阶段1
        d1 = self.dec1_up(e5)  # 32×32, 576
        d1 = torch.cat([d1, e4], dim=1)  # 拼接 e4 (32×32, 384) → 576+384=960
        d1 = self.relu(self.dec1_conv1(d1))
        d1 = self.relu(self.dec1_conv2(d1))  # 32×32, 384

        # 阶段2
        d2 = self.dec2_up(d1)  # 64×64, 288
        d2 = torch.cat([d2, e3], dim=1)  # 拼接 e3 (64×64, 192) → 288+192=480
        d2 = self.relu(self.dec2_conv1(d2))
        d2 = self.relu(self.dec2_conv2(d2))  # 64×64, 288

        # 阶段3
        d3 = self.dec3_up(d2)  # 128×128, 144
        d3 = torch.cat([d3, e2], dim=1)  # 拼接 e2 (128×128, 96) → 144+96=240
        d3 = self.relu(self.dec3_conv1(d3))
        d3 = self.relu(self.dec3_conv2(d3))  # 128×128, 144

        # 阶段4
        d4 = self.dec4_up(d3)  # 256×256, 96
        d4 = torch.cat([d4, e1], dim=1)  # 拼接 e1 (256×256, 48) → 96+48=144
        d4 = self.relu(self.dec4_conv1(d4))
        out = self.dec4_conv2(d4)  # 256×256, 1（输出图像）

        return out


# ---------------------
# 判别器 Discriminator
# ---------------------
class Discriminator(nn.Module):
    def __init__(self, in_channel=1):
        super(Discriminator, self).__init__()
        # 下采样卷积层
        self.conv1 = nn.Conv2d(in_channel, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(768, 1536, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(1536, 3072, kernel_size=3, stride=2, padding=1)

        # 全连接+分类头
        self.fc1 = nn.Linear(3072 * 4 * 4, 3072)
        self.fc2 = nn.Linear(3072, 1)
        self.sigmoid = nn.Sigmoid()

        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # 下采样前向
        x = self.leaky_relu(self.conv1(x))  # 256×256, 48
        x = self.leaky_relu(self.conv2(x))  # 128×128, 96
        x = self.leaky_relu(self.conv3(x))  # 64×64, 192
        x = self.leaky_relu(self.conv4(x))  # 32×32, 384
        x = self.leaky_relu(self.conv5(x))  # 16×16, 768
        x = self.leaky_relu(self.conv6(x))  # 8x8, 1536
        x = self.leaky_relu(self.conv7(x))  # 4×4, 3072
        
        # Flatten + 全连接
        x = x.reshape(x.size(0), -1)  # 展平: batch × (3072×4×4)
        x = self.leaky_relu(self.fc1(x))
        out = self.sigmoid(self.fc2(x))  # 输出0-1置信度

        return out