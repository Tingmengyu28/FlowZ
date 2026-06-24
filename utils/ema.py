import torch


class EMA:
    """
    自定义指数移动平均(EMA)类，适配分布式训练
    核心公式：ema_param = ema_param * decay + model_param * (1 - decay)
    """
    def __init__(self, model, decay=0.999, device=None):
        """
        Args:
            model: 待做EMA的模型
            decay: EMA衰减系数（越大，EMA参数更新越平滑，推荐0.999/0.9999）
            device: EMA参数存放设备（默认和模型一致）
        """
        self.decay = decay
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        
        self.ema_params = {}
        for name, param in model.state_dict().items():
            self.ema_params[name] = param.detach().clone().to(self.device)
        
        self.model_named_params = {k: v for k, v in model.named_parameters()}
        self._saved_model_params = None

    def step(self):
        """
        执行一次EMA更新（需在优化器step后调用）
        仅更新可训练参数，非训练参数（如BN的running_mean）不参与EMA
        """
        with torch.no_grad():  # 禁止梯度计算，节省显存
            for name, param in self.model_named_params.items():
                if param.requires_grad and name in self.ema_params:
                    ema_param = self.ema_params[name]
                    ema_param.mul_(self.decay).add_(param.data.to(self.device), alpha=1 - self.decay)
                    self.ema_params[name] = ema_param

    def store(self):
        """保存当前模型的参数（验证前调用，用于后续恢复）"""
        self._saved_model_params = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }

    def copy_to(self):
        """将EMA参数复制到模型中（验证/推理时调用）"""
        self.model.load_state_dict(self.ema_params, strict=False)

    def restore(self):
        """恢复之前保存的原模型参数（验证后调用，回到训练状态）"""
        if self._saved_model_params is None:
            raise ValueError("请先调用store()保存原模型参数！")
        self.model.load_state_dict(self._saved_model_params, strict=False)
        self._saved_model_params = None  # 清空临时保存的参数

    def state_dict(self):
        """返回EMA参数（用于保存到检查点）"""
        return {k: v.cpu() for k, v in self.ema_params.items()}

    def load_state_dict(self, state_dict):
        """加载EMA参数（断点续训时调用）"""
        for name, param in state_dict.items():
            if name in self.ema_params:
                self.ema_params[name] = param.to(self.device)