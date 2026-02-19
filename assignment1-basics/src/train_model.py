from src.Utils.Training_loop_utils import data_loader, save_checkpoint, load_checkpoint
from src.Utils.utils import AdamW, lr_cosine_schedule, gradient_clipping
from src.Utils.layers import TransformerLM
import numpy as np
import torch
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be disabled.")
def train_model(
model: TransformerLM,
optimizer: AdamW,
num_iters: int,
dataset: np.ndarray,
batch_size: int,
context_length: int,
max_l2_norm: float,
output_path: str,
lr_scheduler: lr_cosine_schedule,
device: str = "cuda",
config: dict = None,
use_wandb: bool = True):
    # 初始化 wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="transformer-lm-training",
            config=config,
            name=f"train_{wandb.util.generate_id()}"
        )
        # 记录模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            "total_params": total_params,
            "trainable_params": trainable_params,
        })
    
    # 使用 tqdm 创建进度条
    pbar = tqdm(range(1, num_iters + 1), desc="Training", unit="iter")
    
    for i in pbar:
        current_lr = lr_scheduler(i)
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
            
        x, y = data_loader(dataset, batch_size, context_length, device=device)
        optimizer.zero_grad()
        
        # 前向传播
        loss = model(x, y)
        
        # 反向传播
        loss.backward()
        
        # 计算梯度范数（裁剪前）
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        
        # 梯度裁剪
        gradient_clipping(model.parameters(), max_l2_norm)
        
        # 计算裁剪后的梯度范数
        total_norm_clipped = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_clipped += param_norm.item() ** 2
        total_norm_clipped = total_norm_clipped ** (1.0 / 2)
        
        optimizer.step()
        
        # 记录指标
        loss_value = loss.item()
        log_dict = {
            "train/loss": loss_value,
            "train/learning_rate": current_lr,
            "train/gradient_norm": total_norm,
            "train/gradient_norm_clipped": total_norm_clipped,
            "train/iteration": i,
        }
        
        # 更新进度条
        pbar.set_postfix({
            "loss": f"{loss_value:.4f}",
            "lr": f"{current_lr:.2e}",
            "grad_norm": f"{total_norm:.2f}"
        })
        
        # 记录到 wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log(log_dict, step=i)
        
        # 保存 checkpoint
        if i % 100 == 0:
            save_checkpoint(model, optimizer, i, output_path)
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({"checkpoint/saved": 1}, step=i)
            tqdm.write(f"Checkpoint saved at iteration {i} to {output_path}")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

# 输出显卡配置信息
if torch.cuda.is_available():
    print("\n=== GPU配置信息 ===")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    print(f"\n当前使用的GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print("=" * 20 + "\n")
else:
    print("未检测到CUDA设备，将使用CPU进行训练")

from src.configs import CS_336_CONFIG, GPT_2_LARGE_CONFIG, GPT_2_SMALL_CONFIG

CONFIG = CS_336_CONFIG
data_path = "output/encoded_data/owt_train_encoded.npy"
output_path = "output/checkpoint/CS_336_OWT_checkpoint.pt"
num_iters = CONFIG["num_iters"]

lr_scheduler = lr_cosine_schedule(CONFIG["max_learning_rate"], CONFIG["min_learning_rate"], CONFIG["warmup_iters"], CONFIG["cosine_cycle_iters"])
model = TransformerLM(CONFIG["vocab_size"], CONFIG["d_model"], CONFIG["num_layers"], CONFIG["num_heads"], CONFIG["d_ff"], CONFIG["rope_theta"], CONFIG["max_seq_len"], device=device)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=CONFIG["max_learning_rate"])

dataset = np.load(data_path, mmap_mode='r')

# 准备配置信息用于 wandb
wandb_config = {
    **CONFIG,
    "num_iters": num_iters,
    "data_path": data_path,
    "output_path": output_path,
    "device": device,
}

train_model(model, optimizer, num_iters, dataset, CONFIG["batch_size"], CONFIG["max_seq_len"], CONFIG["max_l2_norm"], output_path, lr_scheduler, device=device, config=wandb_config, use_wandb=True)