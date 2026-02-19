"""
检查点信息查看脚本
用于查看保存的检查点文件中的相关信息，如迭代次数、模型参数、优化器状态等
"""
import torch
import os
import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path: str):
    """
    查看检查点文件的详细信息
    
    Args:
        checkpoint_path: 检查点文件的路径
    """
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return
    
    print("=" * 60)
    print("检查点信息查看")
    print("=" * 60)
    print(f"\n检查点路径: {checkpoint_path}")
    
    # 获取文件大小
    file_size = os.path.getsize(checkpoint_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.2f} MB ({file_size:,} 字节)")
    
    # 加载检查点
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("\n" + "-" * 60)
        print("检查点内容:")
        print("-" * 60)
        
        # 显示检查点的键
        print(f"\n检查点包含的键: {list(checkpoint.keys())}")
        
        # 迭代次数
        if 'iteration' in checkpoint:
            iteration = checkpoint['iteration']
            print(f"\n✓ 已训练的迭代次数: {iteration}")
        else:
            print("\n✗ 未找到迭代次数信息")
        
        # 模型状态
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"\n✓ 模型状态字典:")
            print(f"  - 参数层数: {len(model_state)}")
            
            # 计算模型参数总数
            total_params = 0
            trainable_params = 0
            param_shapes = {}
            
            for key, value in model_state.items():
                if isinstance(value, torch.Tensor):
                    num_params = value.numel()
                    total_params += num_params
                    trainable_params += num_params  # 假设所有参数都是可训练的
                    param_shapes[key] = list(value.shape)
            
            print(f"  - 总参数数量: {total_params:,}")
            print(f"  - 可训练参数数量: {trainable_params:,}")
            
            # 显示一些参数形状示例（前5个）
            print(f"\n  参数形状示例 (前5个):")
            for i, (key, shape) in enumerate(list(param_shapes.items())):
                print(f"    {key}: {shape}")
        else:
            print("\n✗ 未找到模型状态字典")
        
        # 优化器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer_state = checkpoint['optimizer_state_dict']
            print(f"\n✓ 优化器状态字典:")
            print(f"  - 状态键: {list(optimizer_state.keys())}")
            
            if 'param_groups' in optimizer_state:
                param_groups = optimizer_state['param_groups']
                print(f"  - 参数组数量: {len(param_groups)}")
                
                if len(param_groups) > 0:
                    first_group = param_groups[0]
                    print(f"  - 第一个参数组配置:")
                    for key, value in first_group.items():
                        if key != 'params':  # 跳过params，因为它只是索引列表
                            print(f"    {key}: {value}")
            
            if 'state' in optimizer_state:
                state = optimizer_state['state']
                print(f"  - 优化器状态中的参数数量: {len(state)}")
                
                # 显示第一个参数的状态信息
                if len(state) > 0:
                    first_param_state = list(state.values())[0]
                    print(f"  - 第一个参数的状态键: {list(first_param_state.keys())}")
        else:
            print("\n✗ 未找到优化器状态字典")
        
        # 其他信息
        other_keys = [k for k in checkpoint.keys() if k not in ['iteration', 'model_state_dict', 'optimizer_state_dict']]
        if other_keys:
            print(f"\n其他信息:")
            for key in other_keys:
                print(f"  - {key}: {type(checkpoint[key])}")
        
        print("\n" + "=" * 60)
        print("检查点信息查看完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: 无法加载检查点文件")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 默认检查点路径
    default_checkpoint = "output/checkpoint/GPT_2_LARGE_checkpoint.pt"
    
    # 从命令行参数获取检查点路径，如果没有提供则使用默认路径
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = default_checkpoint
    
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(checkpoint_path):
        # 获取脚本所在目录的父目录（assignment1-basics）
        script_dir = Path(__file__).parent.parent
        checkpoint_path = script_dir / checkpoint_path
    
    inspect_checkpoint(str(checkpoint_path))
