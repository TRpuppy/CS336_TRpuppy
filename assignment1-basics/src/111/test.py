from src.Utils.Training_loop_utils import data_loader
from src.Utils.utils import AdamW
from src.Utils.layers import TransformerLM, TransformerBlock
from src.configs import CS_336_CONFIG
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from jaxtyping import Float, Int
from torch import Tensor

def print_memory_info(stage: str):
    """打印显存使用信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"{stage}:")
        print(f"  已分配显存: {allocated:.3f} GB")
        print(f"  已保留显存: {reserved:.3f} GB")
        print(f"  峰值显存: {max_allocated:.3f} GB")
        print()

def test_checkpoint_memory():
    """测试两种checkpoint方式的显存占用"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA设备，无法测试显存占用")
        return
    
    CONFIG = CS_336_CONFIG
    print(f"配置信息:")
    print(f"  vocab_size: {CONFIG['vocab_size']}")
    print(f"  d_model: {CONFIG['d_model']}")
    print(f"  num_layers: {CONFIG['num_layers']}")
    print(f"  batch_size: {CONFIG['batch_size']}")
    print(f"  max_seq_len: {CONFIG['max_seq_len']}")
    print()
    
    # 准备数据
    data_path = "output/encoded_data/owt_train_encoded.npy"
    try:
        dataset = np.load(data_path, mmap_mode='r')
    except FileNotFoundError:
        print(f"警告: 数据文件 {data_path} 不存在，使用随机数据")
        # 创建随机数据用于测试
        dataset = np.random.randint(0, CONFIG['vocab_size'], size=(100000,), dtype=np.uint16)
    
    # 清空显存缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("=" * 60)
    print("方式1: 块级checkpoint（在TransformerLM中对整个layer设置checkpoint）")
    print("=" * 60)
    
    # 方式1: 块级checkpoint
    class TransformerLM_Method1(TransformerLM):
        """方式1: 在TransformerLM中对整个layer设置checkpoint"""
        def forward(self, x: Int[Tensor, " ... sequence_length"], targets: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, "..."]:
            x = self.token_embeddings(x)
            # 对整个layer设置checkpoint
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
            x = self.rms_norm_final(x)
            x = self.Linear_final(x)
            loss = self.cross_entropy(x, targets)
            return loss
    
    model1 = TransformerLM_Method1(
        CONFIG["vocab_size"], CONFIG["d_model"], CONFIG["num_layers"],
        CONFIG["num_heads"], CONFIG["d_ff"], CONFIG["rope_theta"],
        CONFIG["max_seq_len"], device=device
    ).to(device)
    
    optimizer1 = AdamW(model1.parameters(), lr=CONFIG["max_learning_rate"])
    
    print_memory_info("方式1 - 模型和优化器初始化后")
    
    # 准备一个batch
    x, y = data_loader(dataset, CONFIG["batch_size"], CONFIG["max_seq_len"], device=device)
    
    print_memory_info("方式1 - 数据加载后")
    
    # 前向传播
    optimizer1.zero_grad()
    loss1 = model1(x, y)
    print_memory_info("方式1 - 前向传播后")
    
    # 反向传播
    loss1.backward()
    print_memory_info("方式1 - 反向传播后")
    
    # 记录峰值显存和参数数量（在删除之前）
    peak_memory_method1 = torch.cuda.max_memory_allocated() / 1024**3
    total_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    
    # 清理
    del model1, optimizer1, loss1, x, y
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("\n" + "=" * 60)
    print("方式2: 块内两个checkpoint（在TransformerBlock内部设置两个checkpoint）")
    print("=" * 60)
    
    # 方式2: 块内两个checkpoint
    class TransformerBlock_Method2(TransformerBlock):
        """方式2: 在TransformerBlock内部设置两个checkpoint"""
        def forward(self, x: Float[Tensor, " ... sequence_length d_model"]) -> Float[Tensor, " ... sequence_length d_model"]:
            # 检查点1：MSA前
            y = checkpoint(self._msa_forward, x, use_reentrant=False)
            x = x + y
            
            # 检查点2：SwiGLU前
            y = checkpoint(self._swiglu_forward, x, use_reentrant=False)
            return x + y
    
    class TransformerLM_Method2(TransformerLM):
        """方式2: TransformerBlock内部使用checkpoint，TransformerLM中不使用"""
        def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, max_seq_len: int, device=None, dtype=None):
            # 先调用父类初始化，但我们需要替换layers
            super().__init__(vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta, max_seq_len, device=device, dtype=dtype)
            # 替换所有的TransformerBlock为TransformerBlock_Method2
            self.layers = torch.nn.ModuleList([
                TransformerBlock_Method2(
                    self.d_model, self.num_heads, self.d_ff,
                    self.rope_theta, self.max_seq_len,
                    device=device, dtype=dtype
                ) for _ in range(self.num_layers)
            ])
        
        def forward(self, x: Int[Tensor, " ... sequence_length"], targets: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, "..."]:
            x = self.token_embeddings(x)
            # 不使用checkpoint，因为block内部已经设置了
            for layer in self.layers:
                x = layer(x)
            x = self.rms_norm_final(x)
            x = self.Linear_final(x)
            loss = self.cross_entropy(x, targets)
            return loss
    
    model2 = TransformerLM_Method2(
        CONFIG["vocab_size"], CONFIG["d_model"], CONFIG["num_layers"],
        CONFIG["num_heads"], CONFIG["d_ff"], CONFIG["rope_theta"],
        CONFIG["max_seq_len"], device=device
    ).to(device)
    
    optimizer2 = AdamW(model2.parameters(), lr=CONFIG["max_learning_rate"])
    
    print_memory_info("方式2 - 模型和优化器初始化后")
    
    # 准备一个batch
    x, y = data_loader(dataset, CONFIG["batch_size"], CONFIG["max_seq_len"], device=device)
    
    print_memory_info("方式2 - 数据加载后")
    
    # 前向传播
    optimizer2.zero_grad()
    loss2 = model2(x, y)
    print_memory_info("方式2 - 前向传播后")
    
    # 反向传播
    loss2.backward()
    print_memory_info("方式2 - 反向传播后")
    
    # 记录峰值显存
    peak_memory_method2 = torch.cuda.max_memory_allocated() / 1024**3
    
    # 清理
    del model2, optimizer2, loss2, x, y
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"方式1峰值显存: {peak_memory_method1:.3f} GB")
    print(f"方式2峰值显存: {peak_memory_method2:.3f} GB")
    print(f"显存差异: {abs(peak_memory_method1 - peak_memory_method2):.3f} GB")
    if max(peak_memory_method1, peak_memory_method2) > 0:
        print(f"差异百分比: {abs(peak_memory_method1 - peak_memory_method2) / max(peak_memory_method1, peak_memory_method2) * 100:.2f}%")
    
    # 计算模型参数大小（使用之前保存的total_params）
    param_size_mb = total_params * 4 / 1024**2  # float32, MB
    optimizer_size_mb = total_params * 8 / 1024**2  # AdamW需要2倍参数大小
    print(f"\n模型参数大小: {param_size_mb:.2f} MB")
    print(f"优化器状态大小: {optimizer_size_mb:.2f} MB")
    print(f"参数+优化器总计: {(param_size_mb + optimizer_size_mb) / 1024:.3f} GB")
    print(f"激活值显存（估算）: {peak_memory_method1 - (param_size_mb + optimizer_size_mb) / 1024:.3f} GB (方式1)")

if __name__ == "__main__":
    test_checkpoint_memory()