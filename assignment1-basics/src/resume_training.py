"""
从检查点恢复训练脚本
用于从保存的检查点继续训练模型
"""
from src.Utils.Training_loop_utils import data_loader, save_checkpoint, load_checkpoint
from src.Utils.utils import AdamW, lr_cosine_schedule, gradient_clipping
from src.Utils.layers import TransformerLM
import numpy as np
import torch
from tqdm import tqdm
import os
import sys
from pathlib import Path

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
    use_wandb: bool = True,
    start_iteration: int = 1):
    """
    训练模型（支持从指定迭代次数开始）
    
    Args:
        start_iteration: 开始训练的迭代次数（用于从检查点恢复）
    """
    # 初始化 wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="transformer-lm-training",
            config=config,
            name=f"resume_train_{wandb.util.generate_id()}"
        )
        # 记录模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            "total_params": total_params,
            "trainable_params": trainable_params,
            "resumed_from_iteration": start_iteration
        },allow_val_change=True)
    
    # 使用 tqdm 创建进度条，从start_iteration开始
    pbar = tqdm(range(start_iteration, num_iters + 1), desc="Training", unit="iter")
    
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

def resume_training_from_checkpoint(
    checkpoint_path: str,
    config: dict,
    data_path: str,
    output_path: str,
    device: str = "cuda",
    use_wandb: bool = True):
    """
    从检查点恢复训练
    
    Args:
        checkpoint_path: 检查点文件路径
        config: 模型配置字典
        data_path: 数据文件路径
        output_path: 输出检查点路径
        device: 设备类型
        use_wandb: 是否使用wandb记录
    """
    print("=" * 60)
    print("从检查点恢复训练")
    print("=" * 60)
    
    # 检查检查点文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return
    
    # 加载检查点
    print(f"\n正在加载检查点: {checkpoint_path}")
    
    # 创建模型
    print("\n正在创建模型...")
    model = TransformerLM(
        config["vocab_size"],
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
        config["rope_theta"],
        config["max_seq_len"],
        device=device
    )
    
    # 创建优化器
    print("正在创建优化器...")
    optimizer = AdamW(model.parameters(), lr=config["max_learning_rate"])
    
    # 加载模型和优化器状态
    print("正在加载模型和优化器状态...")
    loaded_iteration = load_checkpoint(checkpoint_path, model, optimizer)
    print(f"成功加载检查点，从迭代 {loaded_iteration} 继续训练")
    
    # 创建学习率调度器
    lr_scheduler = lr_cosine_schedule(
        config["max_learning_rate"],
        config["min_learning_rate"],
        config["warmup_iters"],
        config["cosine_cycle_iters"]
    )
    
    # 加载数据集
    print(f"\n正在加载数据集: {data_path}")
    dataset = np.load(data_path, mmap_mode='r')
    
    # 计算剩余迭代次数
    remaining_iters = config["num_iters"] - loaded_iteration
    if remaining_iters <= 0:
        print(f"\n警告: 检查点显示已训练 {loaded_iteration} 次迭代，已达到目标 {config['num_iters']} 次迭代")
        print("训练已完成，无需继续训练")
        return
    
    print(f"\n剩余迭代次数: {remaining_iters}")
    print(f"目标总迭代次数: {config['num_iters']}")
    
    # 准备配置信息用于 wandb
    wandb_config = {
        **config,
        "num_iters": config["num_iters"],
        "data_path": data_path,
        "output_path": output_path,
        "device": device,
        "resumed_from_checkpoint": checkpoint_path,
        "resumed_from_iteration": loaded_iteration,
    }
    
    print("\n" + "=" * 60)
    print("开始恢复训练")
    print("=" * 60 + "\n")
    
    model = model.to(device)
    train_model(
        model,
        optimizer,
        config["num_iters"],
        dataset,
        config["batch_size"],
        config["max_seq_len"],
        config["max_l2_norm"],
        output_path,
        lr_scheduler,
        device=device,
        config=wandb_config,
        use_wandb=use_wandb,
        start_iteration=loaded_iteration + 1  # 从下一个迭代开始
    )
    
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
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
    from src.configs import GPT_2_LARGE_CONFIG, CS_336_CONFIG
    # 默认配置和路径
    CONFIG = CS_336_CONFIG
    checkpoint_path = "output/checkpoint/CS_336_OWT_checkpoint.pt"
    data_path = "output/encoded_data/owt_train_encoded.npy"
    output_path = "output/checkpoint/CS_336_OWT_checkpoint.pt"
    
    # 从命令行参数获取检查点路径
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(checkpoint_path):
        script_dir = Path(__file__).parent.parent
        checkpoint_path = script_dir / checkpoint_path
        data_path = script_dir / data_path
        output_path = script_dir / output_path
    
    # 开始恢复训练
    resume_training_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config=CONFIG,
        data_path=str(data_path),
        output_path=str(output_path),
        device=device,
        use_wandb=True
    )
