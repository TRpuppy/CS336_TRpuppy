"""
验证脚本
读取checkpoint，在验证数据集上评估平均loss
"""
from src.Utils.Training_loop_utils import load_checkpoint
from src.Utils.utils import AdamW
from src.Utils.layers import TransformerLM
import numpy as np
import torch
from tqdm import tqdm
import os
import sys
from pathlib import Path

def validate_model(
    checkpoint_path: str,
    config: dict,
    valid_data_path: str,
    device: str = "cuda",
    batch_size: int = 32):
    """
    在验证数据集上评估模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        config: 模型配置字典
        valid_data_path: 验证数据文件路径
        device: 设备类型
        batch_size: 批处理大小（用于加速评估）
    """
    print("=" * 60)
    print("模型验证")
    print("=" * 60)
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: checkpoint文件不存在: {checkpoint_path}")
        return
    
    # 检查验证数据文件是否存在
    if not os.path.exists(valid_data_path):
        print(f"错误: 验证数据文件不存在: {valid_data_path}")
        return
    
    # 加载checkpoint
    print(f"\n正在加载checkpoint: {checkpoint_path}")
    
    # 创建模型
    print("正在创建模型...")
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
    
    # 创建优化器（仅用于加载checkpoint，不会用于训练）
    print("正在创建优化器...")
    optimizer = AdamW(model.parameters(), lr=config["max_learning_rate"])
    
    # 加载模型状态
    print("正在加载模型状态...")
    loaded_iteration = load_checkpoint(checkpoint_path, model, optimizer)
    print(f"成功加载checkpoint，迭代次数: {loaded_iteration}")
    
    # 将模型设置为评估模式
    model.eval()
    model = model.to(device)
    
    # 加载验证数据集
    print(f"\n正在加载验证数据集: {valid_data_path}")
    valid_data = np.load(valid_data_path, mmap_mode='r')
    print(f"验证数据集大小: {len(valid_data)} tokens")
    
    # 将数据切成若干个长度为max_seq_len的串
    max_seq_len = config["max_seq_len"]
    # 确保y也能完整提取，所以需要len(valid_data) >= max_seq_len + 1
    # 每个segment需要max_seq_len+1个token（x需要max_seq_len，y需要max_seq_len，但y是x的shift）
    num_segments = (len(valid_data) - 1) // max_seq_len
    print(f"\n将数据切成 {num_segments} 个长度为 {max_seq_len} 的串")
    print(f"总数据长度: {len(valid_data)} tokens")
    print(f"每个串需要: {max_seq_len} tokens (x) + 1 token (y的最后一个)")
    
    # 计算所有串上的平均loss
    total_loss = 0.0
    num_batches = 0
    
    # 使用批处理加速评估
    with torch.no_grad():
        # 创建进度条
        pbar = tqdm(range(0, num_segments, batch_size), desc="评估中", unit="batch")
        
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, num_segments)
            batch_segments = batch_end - batch_start
            
            # 准备批处理数据
            x_batch = []
            y_batch = []
            
            for i in range(batch_start, batch_end):
                start_idx = i * max_seq_len
                end_idx = start_idx + max_seq_len
                
                # 提取x和y（y是x向右shift 1位）
                x = valid_data[start_idx:end_idx]
                y = valid_data[start_idx + 1:end_idx + 1]
                
                x_batch.append(x)
                y_batch.append(y)
            
            # 如果批次为空，跳过
            if len(x_batch) == 0:
                continue
            
            # 转换为tensor
            x_tensor = torch.from_numpy(np.array(x_batch, dtype=np.uint16)).to(device)
            y_tensor = torch.from_numpy(np.array(y_batch, dtype=np.uint16)).to(device)
            
            # 前向传播计算loss
            loss = model(x_tensor, y_tensor)
            
            # 累加loss（使用实际处理的segment数量）
            actual_segments = len(x_batch)
            total_loss += loss.item() * actual_segments
            num_batches += actual_segments
            
            # 更新进度条
            current_avg_loss = total_loss / num_batches
            pbar.set_postfix({
                "avg_loss": f"{current_avg_loss:.4f}",
                "segments": f"{num_batches}/{num_segments}"
            })
    
    # 计算平均loss
    avg_loss = total_loss / num_batches
    
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"评估的串数量: {num_batches}")
    print(f"每个串的长度: {max_seq_len}")
    print(f"平均loss: {avg_loss:.6f}")
    print("=" * 60)
    
    return avg_loss

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
        print(f"\n当前使用的GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print("=" * 20 + "\n")
    else:
        print("未检测到CUDA设备，将使用CPU进行评估")
    
    # 配置（与训练脚本保持一致）
    from src.configs import CS_336_CONFIG, GPT_2_LARGE_CONFIG, GPT_2_SMALL_CONFIG
    
    # 默认配置和路径
    CONFIG = CS_336_CONFIG
    checkpoint_path = "output/checkpoint/CS_336_OWT_checkpoint.pt"
    valid_data_path = "output/encoded_data/owt_valid_encoded.npy"
    
    # 从命令行参数获取checkpoint路径
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    # 从命令行参数获取验证数据路径
    if len(sys.argv) > 2:
        valid_data_path = sys.argv[2]
    
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(checkpoint_path):
        script_dir = Path(__file__).parent.parent
        checkpoint_path = script_dir / checkpoint_path
        valid_data_path = script_dir / valid_data_path
    
    # 开始验证
    validate_model(
        checkpoint_path=str(checkpoint_path),
        config=CONFIG,
        valid_data_path=str(valid_data_path),
        device=device,
        batch_size=32  # 可以根据GPU内存调整
    )
