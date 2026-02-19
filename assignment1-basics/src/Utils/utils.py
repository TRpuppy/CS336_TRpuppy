import torch
from collections.abc import Iterable, Callable
from typing import Optional, Tuple
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter],
    lr: float = 1e-3, 
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2):
        defaults = {"lr":lr, "beta1":betas[0], "beta2":betas[1], "weight_decay":weight_decay}
        self.eps = eps
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("step", 1)
                momentum = state.get("momentum", torch.zeros_like(p.data))
                velocity = state.get("velocity", torch.zeros_like(p.data))
                grad = p.grad.data
                momentum = beta1 * momentum + (1 - beta1) * grad
                velocity = beta2 * velocity + (1 - beta2) * grad**2
                momentum_corrected = momentum / (1 - beta1**t)
                velocity_corrected = velocity / (1 - beta2**t)
                p.data -= lr * momentum_corrected / (torch.sqrt(velocity_corrected) + self.eps)
                p.data = p.data * (1 - lr * weight_decay)
                state["step"] = t + 1
                state["momentum"] = momentum
                state["velocity"] = velocity

        return loss

class lr_cosine_schedule:
    def __init__(self, 
    max_learning_rate: float, 
    min_learning_rate: float, 
    warmup_iters: int, 
    cosine_cycle_iters: int):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def __call__(self, it: int) -> float:
        if it < self.warmup_iters:
            return self.max_learning_rate * it / self.warmup_iters
        elif it <= self.cosine_cycle_iters:
            return self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * \
            (1 + math.cos(math.pi * (it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters))) / 2
        else:
            return self.min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # 计算所有参数梯度的全局L2范数
    parameters = list(parameters)
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    
    # 如果全局范数超过阈值，按比例缩放所有梯度
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)