import cv2
import os
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from typing import Tuple


class DarkSectioning:
    def __init__(
        self,
        block_size: int = 15,
        lowpass_radius: int = 30,
        iter_num: int = 5,
        dark_thresh_percentile: int = 95,
        eps: float = 1e-6
    ):
        """
        初始化Dark Sectioning参数（严格遵循论文推荐值）
        Args:
            block_size: 暗通道计算块大小（略大于聚焦PSF，论文推荐15-25）
            lowpass_radius: 双频分离低通半径（控制低频背景范围，论文推荐20-40）
            iter_num: 背景迭代优化次数（论文推荐3-10，平衡效果与速度）
            dark_thresh_percentile: 暗通道背景阈值（论文用前5%高值，即95分位数）
            eps: 数值稳定性参数（避免除以零）
        """
        self.block_size = block_size
        self.lowpass_radius = lowpass_radius
        self.iter_num = iter_num
        self.dark_thresh_percentile = dark_thresh_percentile
        self.eps = eps

    def _dark_channel(self, img: np.ndarray) -> np.ndarray:
        """
        计算暗通道（论文核心：聚焦区域暗通道≈0，离焦背景暗通道≠0）
        Args:
            img: 归一化后的单通道WF图像 (H, W)，值域[0,1]
        Returns:
            dark_ch: 暗通道图像 (H, W)，值域[0,1]
        """
        # 论文定义：局部块内取最小值（用腐蚀操作实现，结构元素为矩形）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.block_size, self.block_size))
        dark_ch = cv2.erode(img, kernel)
        return dark_ch

    def _dual_frequency_separation(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        双频分离（论文核心：高频保留聚焦信号，低频处理离焦背景）
        Args:
            img: 归一化后的单通道WF图像 (H, W)，值域[0,1]
        Returns:
            img_high: 高频部分（聚焦信号，直接保留）(H, W)，值域[0,1]
            img_low: 低频部分（离焦背景，需处理）(H, W)，值域[0,1]
        """
        H, W = img.shape
        # 1. 频域转换（FFT+中心化）
        fft_img = fft2(img)
        fft_shift = fftshift(fft_img)

        # 2. 构建高斯低通滤波器（论文用高斯核分离低频背景）
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        center = (H // 2, W // 2)
        dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        lowpass_filter = np.exp(-(dist ** 2) / (2 * self.lowpass_radius ** 2))

        # 3. 分离高低频（高频=原始-低频，避免信号丢失）
        fft_low = fft_shift * lowpass_filter  # 低频（背景）
        fft_high = fft_shift * (1 - lowpass_filter)  # 高频（聚焦信号）

        # 4. 逆变换回空域（仅去虚部，不全局归一化，保留原始强度）
        img_low = np.real(ifft2(ifftshift(fft_low)))
        img_high = np.real(ifft2(ifftshift(fft_high)))

        # 数值裁剪（避免极端值，不拉伸，符合论文“保留信号强度”原则）
        img_low = np.clip(img_low, 0.0, 1.0)
        img_high = np.clip(img_high, 0.0, 1.0)
        return img_high, img_low

    def _estimate_background(self, dark_ch: np.ndarray, img_low: np.ndarray) -> np.ndarray:
        """
        估计不均匀背景A(x,y)（论文核心：适配荧光图像非均匀背景，无固定大气光）
        Args:
            dark_ch: 暗通道图像 (H, W)，值域[0,1]
            img_low: 双频分离后的低频部分 (H, W)，值域[0,1]
        Returns:
            A: 不均匀背景图像 (H, W)，值域[0,1]
        """
        # 1. 基于暗通道选择背景区域（论文：暗通道高值对应离焦背景）
        dark_thresh = np.percentile(dark_ch, self.dark_thresh_percentile) if np.any(dark_ch > 0) else 0.1
        background_mask = (dark_ch >= dark_thresh).astype(np.float32)

        # 2. 膨胀背景掩码（确保覆盖完整背景区域，论文用小核膨胀）
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
        background_mask = cv2.dilate(background_mask, dilate_kernel)

        # 3. 加权平均估计背景（背景区域权重高，避免前景信号干扰）
        # 论文公式：A = (filter2D(img_low*mask) / filter2D(mask))
        A = cv2.filter2D(img_low * background_mask, -1, dilate_kernel)
        A = A / (cv2.filter2D(background_mask, -1, dilate_kernel) + self.eps)

        # 4. 填补背景外区域（论文用inpaint修复非背景区域的背景值）
        # OpenCV inpaint要求：输入为8位单通道，掩码为255标记修复区域
        A_uint8 = (A * 255).astype(np.uint8)
        inpaint_mask = (background_mask == 0).astype(np.uint8) * 255  # 非背景区域需填补
        A_inpainted = cv2.inpaint(A_uint8, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # 5. 转回归一化浮点数（符合后续计算需求）
        A = A_inpainted / 255.0
        # 限制背景最大值（论文：避免背景估计过高，默认≤0.6）
        A = np.clip(A, 0.0, 0.6)
        return A

    def _iterative_transmission(self, img_low: np.ndarray, A: np.ndarray, dark_ch: np.ndarray) -> np.ndarray:
        """
        迭代优化传输比t(x,y)（论文核心：局部+全局背景迭代去除）
        Args:
            img_low: 双频分离后的低频部分 (H, W)，值域[0,1]
            A: 不均匀背景图像 (H, W)，值域[0,1]
            dark_ch: 暗通道图像 (H, W)，值域[0,1]
        Returns:
            t: 传输比图像（聚焦信号占比）(H, W)，值域[0.1,1.0]
        """
        t = np.ones_like(img_low)  # 初始传输比=1（无背景）
        for _ in range(self.iter_num):
            # 论文公式：t = 1 - (dark_ch / (A + eps))（背景越浓，t越小）
            t = 1 - (dark_ch / (A + self.eps))
            # 限制传输比范围（论文：避免t过小导致前景过度抑制，范围[0.1,1.0]）
            t = np.clip(t, 0.05, 1.0)
            # 迭代更新背景（论文：基于当前传输比优化背景估计）
            A = cv2.GaussianBlur(img_low * (1 - t) + A * t, (5, 5), 0)
            A = np.clip(A, 0.0, 0.6)  # 再次限制背景最大值
        return t

    def __call__(self, wf_img: np.ndarray) -> np.ndarray:
        """
        主函数：输入WF图像，输出类Confocal效果图像（论文完整流程）
        Args:
            wf_img: 输入单通道WF图像 (H, W)，值域[0,255]
        Returns:
            confocal_like_img: 输出类Confocal图像 (H, W)，值域[0,255]
        """
        # 1. 输入校验（严格单通道，符合论文荧光图像输入要求）
        if len(wf_img.shape) != 2:
            raise ValueError(f"输入必须是单通道WF图像！当前形状：{wf_img.shape}")
        if wf_img.dtype != np.uint8:
            raise ValueError(f"输入必须是uint8类型！当前类型：{wf_img.dtype}")

        # 2. 预处理：归一化到[0,1]（保留原始明暗关系，论文不建议拉伸）
        img_norm = wf_img / 255.0
        original_max = img_norm.max()  # 记录原始最大强度，用于后续校准

        # 3. 核心步骤1：计算暗通道（区分聚焦/离焦）
        dark_ch = self._dark_channel(img_norm)

        # 4. 核心步骤2：双频分离（保护高频聚焦信号）
        img_high, img_low = self._dual_frequency_separation(img_norm)

        # 5. 核心步骤3：估计不均匀背景（适配荧光图像）
        A = self._estimate_background(dark_ch, img_low)

        # 6. 核心步骤4：迭代优化传输比（去除低频背景）
        t = self._iterative_transmission(img_low, A, dark_ch)

        # 7. 核心步骤5：低频背景去除（论文公式：J_Lo = (I_Lo - (1-t)*A) / t）
        img_low_processed = (img_low - (1 - t) * A) / (t + self.eps)
        img_low_processed = np.clip(img_low_processed, 0.0, original_max)  # 按原始强度校准

        # 8. 核心步骤6：高低频合并（论文：高频直接保留，低频处理后合并）
        # 高频权重0.5，低频权重0.5（论文推荐，平衡细节与背景去除）
        result_norm = img_high * 0.5 + img_low_processed * 0.5
        result_norm = np.clip(result_norm, 0.0, original_max)  # 避免强度溢出

        # 9. 后处理：转回uint8（单通道输出，符合PNG格式要求）
        confocal_like_img = (result_norm * 255).astype(np.uint8)
        return confocal_like_img


# ---------------------- 论文效果验证：WF→类Confocal处理流程 ----------------------
def wf_to_confocal_like(
    input_png_path: str,
    output_png_path: str,
    block_size: int = 15,
    lowpass_radius: int = 30,
    iter_num: int = 5,
    dark_thresh_percentile: int = 95,
) -> None:
    """
    完整流程：读取WF-PNG→Dark Sectioning→保存类Confocal-PNG
    Args:
        input_png_path: 输入WF图像路径（单通道PNG）
        output_png_path: 输出类Confocal图像路径（单通道PNG）
        block_size: 暗通道块大小（论文推荐15-25）
        lowpass_radius: 低通半径（论文推荐20-40）
        iter_num: 迭代次数（论文推荐3-10）
    """
    # 1. 读取单通道WF图像（论文：荧光图像为单通道灰度图）
    wf_img = cv2.imread(input_png_path, cv2.IMREAD_GRAYSCALE)
    if wf_img is None:
        raise FileNotFoundError(f"未找到输入图像：{input_png_path}，请检查路径")

    # 2. 初始化Dark Sectioning（参数符合论文推荐）
    dark_section = DarkSectioning(
        block_size=block_size,
        lowpass_radius=lowpass_radius,
        iter_num=iter_num,
        dark_thresh_percentile=dark_thresh_percentile,
    )

    # 3. 执行处理（WF→类Confocal）
    confocal_like_img = dark_section(wf_img)
    confocal_like_img = (confocal_like_img - confocal_like_img.min()) / (confocal_like_img.max() - confocal_like_img.min()) * 255

    # 4. 保存输出（单通道PNG，符合论文结果格式）
    cv2.imwrite(output_png_path, confocal_like_img)
    print(f"处理完成！类Confocal图像已保存至：{output_png_path}")

# ---------------------- 示例：调用函数处理WF图像 ----------------------
if __name__ == "__main__":

    data_name = "real_slope"
    model_name = "fm_palette"
    image_idx = "1"
    # 论文效果复现参数（基于论文Fig.1/3的最优参数）
    WF_PNG_PATH = f"/data1/azt/cv/recoverZ/outputs/{data_name}/{model_name}/inference_details/pred/{image_idx}.png"       # 输入：单通道WF-PNG（如tubulin、神经元图像）
    os.makedirs(f"/data1/azt/cv/recoverZ/outputs/{data_name}/{model_name}/inference_details/pred_post", exist_ok=True)
    OUTPUT_PNG_PATH = f"/data1/azt/cv/recoverZ/outputs/{data_name}/{model_name}/inference_details/pred_post/{image_idx}.png"    # 输出：类Confocal-PNG

    # 调用处理（基于论文推荐的参数）
    wf_to_confocal_like(
        input_png_path=WF_PNG_PATH,
        output_png_path=OUTPUT_PNG_PATH,
        block_size=20,
        lowpass_radius=30,
        iter_num=5,
        dark_thresh_percentile=95,
    )
