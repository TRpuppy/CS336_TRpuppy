import torch
import torch.nn as nn

# 测试1: a和b是绝对值（-3*std, 3*std）
std1 = 0.5
tensor1 = torch.empty(1000, 1000)
nn.init.trunc_normal_(tensor1, mean=0.0, std=std1, a=-3*std1, b=3*std1)
print(f'测试1 (a=-3*std, b=3*std, std={std1}):')
print(f'  最小值: {tensor1.min().item():.4f}, 最大值: {tensor1.max().item():.4f}')
print(f'  理论范围: [{-3*std1:.4f}, {3*std1:.4f}]')
print()

# 测试2: a和b是固定值（-3, 3）
std2 = 0.5
tensor2 = torch.empty(1000, 1000)
nn.init.trunc_normal_(tensor2, mean=0.0, std=std2, a=-3, b=3)
print(f'测试2 (a=-3, b=3, std={std2}):')
print(f'  最小值: {tensor2.min().item():.4f}, 最大值: {tensor2.max().item():.4f}')
print(f'  理论范围: [-3.0000, 3.0000]')
print()

# 测试3: 不同std，使用-3*std和3*std
std3 = 1.0
tensor3 = torch.empty(1000, 1000)
nn.init.trunc_normal_(tensor3, mean=0.0, std=std3, a=-3*std3, b=3*std3)
print(f'测试3 (a=-3*std, b=3*std, std={std3}):')
print(f'  最小值: {tensor3.min().item():.4f}, 最大值: {tensor3.max().item():.4f}')
print(f'  理论范围: [{-3*std3:.4f}, {3*std3:.4f}]')