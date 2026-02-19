"""
使用torch.profiler进行详细的内存分析
分析前向传播和反向传播中模型每一层的内存占用
"""
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Dict, List, Optional, Tuple
import json
from collections import defaultdict


class LayerMemoryProfiler:
    """逐层内存分析器"""
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.layer_memory_stats: Dict[str, Dict] = {}
        self.hooks = []
        
    def _register_hooks(self):
        """注册前向和反向传播的hook来追踪每一层"""
        def make_hook(name):
            def forward_hook(module, input, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
                    
                    if name not in self.layer_memory_stats:
                        self.layer_memory_stats[name] = {
                            "forward_allocated": [],
                            "forward_reserved": [],
                            "backward_allocated": [],
                            "backward_reserved": [],
                        }
                    
                    self.layer_memory_stats[name]["forward_allocated"].append(allocated)
                    self.layer_memory_stats[name]["forward_reserved"].append(reserved)
            
            def backward_hook(module, grad_input, grad_output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
                    
                    if name in self.layer_memory_stats:
                        self.layer_memory_stats[name]["backward_allocated"].append(allocated)
                        self.layer_memory_stats[name]["backward_reserved"].append(reserved)
            
            return forward_hook, backward_hook
        
        # 为每个子模块注册hook
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 只注册叶子模块
                forward_hook, backward_hook = make_hook(name)
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
    
    def _remove_hooks(self):
        """移除所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def profile_with_profiler(self, x, y, output_dir: str = "memory_profiles"):
        """使用torch.profiler进行详细的内存分析"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 重置内存统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        print("\n开始内存分析...")
        print("="*80)
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            # 前向传播 - 直接调用模型，模型内部已经有record_function标记
            with record_function("forward_pass"):
                loss = self.model(x, y)
            
            # 反向传播
            with record_function("backward_pass"):
                loss.backward()
        
        # 保存详细报告
        print("\n生成内存分析报告...")
        
        # 1. 控制台输出（按内存使用排序）
        print("\n" + "="*80)
        print("内存使用分析（按CUDA内存使用排序）")
        print("="*80)
        print(prof.key_averages().table(
            sort_by="cuda_memory_usage",
            row_limit=50,
            max_name_column_width=60
        ))
        
        # 2. 按操作类型分组统计
        print("\n" + "="*80)
        print("按操作类型分组的内存统计")
        print("="*80)
        
        # 收集所有事件
        events = prof.key_averages()
        grouped_stats = defaultdict(lambda: {
            "count": 0,
            "total_cuda_memory": 0,
            "total_cpu_memory": 0,
            "total_cuda_time": 0,
            "total_cpu_time": 0,
        })
        
        for event in events:
            key = event.key
            # 提取操作类型（去掉具体参数）
            op_type = key.split("(")[0].strip()
            
            grouped_stats[op_type]["count"] += 1
            if hasattr(event, "cuda_memory_usage"):
                grouped_stats[op_type]["total_cuda_memory"] += event.cuda_memory_usage
            if hasattr(event, "cpu_memory_usage"):
                grouped_stats[op_type]["total_cpu_memory"] += event.cpu_memory_usage
            if hasattr(event, "cuda_time"):
                grouped_stats[op_type]["total_cuda_time"] += event.cuda_time
            if hasattr(event, "cpu_time"):
                grouped_stats[op_type]["total_cpu_time"] += event.cpu_time
        
        # 打印分组统计
        print(f"{'操作类型':<40} {'次数':<10} {'CUDA内存(MB)':<15} {'CPU内存(MB)':<15}")
        print("-"*80)
        sorted_ops = sorted(grouped_stats.items(), 
                          key=lambda x: x[1]["total_cuda_memory"], 
                          reverse=True)
        for op_type, stats in sorted_ops[:20]:  # 只显示前20个
            cuda_mem_mb = stats["total_cuda_memory"] / (1024**2)
            cpu_mem_mb = stats["total_cpu_memory"] / (1024**2)
            print(f"{op_type:<40} {stats['count']:<10} {cuda_mem_mb:<15.2f} {cpu_mem_mb:<15.2f}")
        
        # 3. 按层分组统计 - 使用事件树结构
        print("\n" + "="*80)
        print("按层分组的内存统计")
        print("="*80)
        
        layer_stats = defaultdict(lambda: {
            "forward_cuda_memory": 0,
            "backward_cuda_memory": 0,
            "forward_cpu_memory": 0,
            "backward_cpu_memory": 0,
        })
        
        # 定义要追踪的层名称模式
        layer_patterns = [
            "token_embeddings",
            "final_rms_norm",
            "final_linear",
            "cross_entropy",
        ]
        
        # 遍历所有事件，查找record_function标记
        for event in events:
            key = event.key
            
            # 跳过aten::开头的底层操作
            if key.startswith("aten::"):
                continue
            
            # 检查是否是record_function标记的层
            layer_name = None
            
            # 检查是否是transformer层
            if "layer_" in key:
                parts = key.split("_")
                if len(parts) >= 2 and parts[0] == "layer" and parts[1].isdigit():
                    layer_num = parts[1]
                    if len(parts) > 2:
                        # 有操作后缀，如 layer_0_rms_norm1
                        operation = "_".join(parts[2:])
                        layer_name = f"layer_{layer_num}_{operation}"
                    else:
                        # 整个层
                        layer_name = f"layer_{layer_num}"
            
            # 检查是否是其他标记的层
            for pattern in layer_patterns:
                if pattern in key:
                    layer_name = pattern
                    break
            
            if layer_name:
                # 判断是前向还是反向（通过检查key中是否包含backward）
                is_backward = "backward" in key.lower() or "grad" in key.lower()
                
                # 获取内存使用
                cuda_mem = getattr(event, "cuda_memory_usage", 0) or 0
                cpu_mem = getattr(event, "cpu_memory_usage", 0) or 0
                
                if cuda_mem > 0:
                    if is_backward:
                        layer_stats[layer_name]["backward_cuda_memory"] += cuda_mem
                    else:
                        layer_stats[layer_name]["forward_cuda_memory"] += cuda_mem
                
                if cpu_mem > 0:
                    if is_backward:
                        layer_stats[layer_name]["backward_cpu_memory"] += cpu_mem
                    else:
                        layer_stats[layer_name]["forward_cpu_memory"] += cpu_mem
        
        # 打印层统计
        print(f"{'层名称':<30} {'前向CUDA(MB)':<15} {'反向CUDA(MB)':<15} {'总计CUDA(MB)':<15}")
        print("-"*80)
        for layer_key, stats in sorted(layer_stats.items()):
            forward_mb = stats["forward_cuda_memory"] / (1024**2)
            backward_mb = stats["backward_cuda_memory"] / (1024**2)
            total_mb = (stats["forward_cuda_memory"] + stats["backward_cuda_memory"]) / (1024**2)
            print(f"{layer_key:<30} {forward_mb:<15.2f} {backward_mb:<15.2f} {total_mb:<15.2f}")
        
        # 4. 导出Chrome trace格式（可在chrome://tracing中查看）
        trace_file = f"{output_dir}/memory_trace.json"
        prof.export_chrome_trace(trace_file)
        print(f"\nChrome trace文件已保存到: {trace_file}")
        print("可以在浏览器中打开 chrome://tracing 并加载此文件查看详细的时间线")
        
        # 5. 导出JSON格式的详细数据
        json_file = f"{output_dir}/memory_stats.json"
        export_data = {
            "total_events": len(events),
            "grouped_by_operation": {
                op: {
                    "count": stats["count"],
                    "total_cuda_memory_mb": stats["total_cuda_memory"] / (1024**2),
                    "total_cpu_memory_mb": stats["total_cpu_memory"] / (1024**2),
                }
                for op, stats in grouped_stats.items()
            },
            "grouped_by_layer": {
                layer: {
                    "forward_cuda_memory_mb": stats["forward_cuda_memory"] / (1024**2),
                    "backward_cuda_memory_mb": stats["backward_cuda_memory"] / (1024**2),
                    "total_cuda_memory_mb": (stats["forward_cuda_memory"] + stats["backward_cuda_memory"]) / (1024**2),
                }
                for layer, stats in layer_stats.items()
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"JSON统计文件已保存到: {json_file}")
        
        return prof, export_data


def analyze_layer_memory_detailed(model, x, y, device="cuda", output_dir="memory_profiles"):
    """
    详细分析每一层的内存占用
    
    Args:
        model: TransformerLM模型
        x: 输入tokens
        y: 目标tokens
        device: 设备
        output_dir: 输出目录
    """
    profiler = LayerMemoryProfiler(model, device)
    prof, stats = profiler.profile_with_profiler(x, y, output_dir)
    return prof, stats
