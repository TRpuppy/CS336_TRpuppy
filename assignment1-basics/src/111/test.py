import sys
import time
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.BPE_Tokenizer.BPE import BPE

def main():
    # 设置文件路径
    vocab_path = project_root / "output" / "vocab_tiny_train.pkl"
    merges_path = project_root / "output" / "merges_tiny_train.pkl"
    data_path = project_root / "data" / "TinyStoriesV2-GPT4-valid.txt"
    
    # 检查文件是否存在
    if not vocab_path.exists():
        print(f"Error: {vocab_path} not found")
        return
    if not merges_path.exists():
        print(f"Error: {merges_path} not found")
        return
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return
    
    # 加载 BPE tokenizer
    print("Loading BPE tokenizer...")
    tokenizer = BPE.from_files(
        str(vocab_path),
        str(merges_path),
        special_tokens=["<|endoftext|>"]
    )
    print("BPE tokenizer loaded successfully!")
    
    # 目标读取大小：约5MB (5 * 1024 * 1024 字节)
    target_size = 1 * 1024 * 1024  # 5MB
    
    print(f"Reading approximately {target_size / (1024*1024):.1f}MB from {data_path}...")
    
    # 读取约5MB的字符串（作为一个整体）
    large_string = ""
    total_bytes = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        with tqdm(total=target_size, desc="Reading", unit="B", unit_scale=True) as pbar:
            while total_bytes < target_size:
                chunk = f.read(64 * 1024)  # 每次读取64KB
                if not chunk:
                    break
                
                chunk_bytes = len(chunk.encode('utf-8'))
                
                # 如果加上这个chunk会超过目标大小，截取到目标大小
                if total_bytes + chunk_bytes > target_size:
                    remaining_bytes = target_size - total_bytes
                    # 估算需要截取的字符数
                    remaining_chars = int(remaining_bytes / 2)  # 保守估计
                    chunk = chunk[:remaining_chars]
                    chunk_bytes = len(chunk.encode('utf-8'))
                
                large_string += chunk
                total_bytes += chunk_bytes
                pbar.update(chunk_bytes)
                
                if total_bytes >= target_size:
                    break
    
    print(f"\nRead {total_bytes / (1024*1024):.2f}MB string (length: {len(large_string):,} characters)")
    print(f"Starting encoding with encode() function...")
    
    # 测量编码时间
    start_time = time.time()
    
    # 使用tqdm显示处理状态（虽然无法显示精确进度，但可以显示正在处理）
    with tqdm(desc="Encoding", bar_format='{desc}: {elapsed}') as pbar:
        tokens = tokenizer.encode(large_string)
        pbar.update(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Encoding completed!")
    print(f"{'='*60}")
    print(f"Input size: {total_bytes / (1024*1024):.2f}MB ({len(large_string):,} characters)")
    print(f"Output tokens: {len(tokens):,}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Processing speed: {total_bytes / (1024*1024) / elapsed_time:.2f} MB/s")
    print(f"Token generation rate: {len(tokens) / elapsed_time:,.0f} tokens/s")
    if total_bytes > 0:
        print(f"Average tokens per MB: {len(tokens) / (total_bytes / (1024*1024)):.2f}")

if __name__ == "__main__":
    main()
