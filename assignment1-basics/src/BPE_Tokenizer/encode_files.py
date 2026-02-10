import os
import pickle
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, BinaryIO
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import numpy as np

from .BPE import BPE


def find_chunk_boundaries_1mb(
    file: BinaryIO,
    chunk_size_mb: float = 1.0,
    split_special_token: bytes = b"<|endoftext|>",
) -> List[int]:
    """
    将文件分成约1MB的块，在special token处分割
    
    Args:
        file: 文件句柄（BinaryIO）
        chunk_size_mb: 目标块大小（MB）
        split_special_token: 用于分割的特殊token
    
    Returns:
        块边界位置列表（字节位置）
    """
    # 获取文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
    
    # 计算期望的块数
    desired_num_chunks = max(1, file_size // chunk_size_bytes)
    
    # 使用类似 Pre_Tokenizer 的方法
    chunk_boundaries = [0]
    mini_chunk_size = 4096  # 每次读取4KB
    
    current_pos = 0
    while current_pos < file_size:
        next_target = min(current_pos + chunk_size_bytes, file_size)
        
        if next_target >= file_size:
            chunk_boundaries.append(file_size)
            break
        
        # 在目标位置附近查找special token
        # 向前搜索4KB，向后搜索4KB
        search_start = max(current_pos, next_target - mini_chunk_size)
        search_end = min(next_target + mini_chunk_size, file_size)
        file.seek(search_start)
        
        found = False
        search_pos = search_start
        buffer = b''
        
        while search_pos < search_end:
            remaining = min(mini_chunk_size, search_end - search_pos)
            mini_chunk = file.read(remaining)
            
            if not mini_chunk:
                break
            
            buffer += mini_chunk
            search_pos += len(mini_chunk)
            
            # 在buffer中查找special token
            token_pos = buffer.find(split_special_token)
            if token_pos != -1:
                # 找到了special token
                actual_pos = search_start + token_pos + len(split_special_token)
                chunk_boundaries.append(actual_pos)
                current_pos = actual_pos
                found = True
                break
        
        if not found:
            # 没找到special token，就在目标位置分割
            chunk_boundaries.append(next_target)
            current_pos = next_target
    
    return sorted(set(chunk_boundaries))


def encode_chunk(
    file_path: str,
    start: int,
    end: int,
    vocab_path: str,
    merges_path: str,
    special_tokens: List[str],
    temp_dir: str,
    chunk_id: int,
) -> str:
    """
    编码一个文件块
    
    Args:
        file_path: 原始文件路径
        start: 块起始位置（字节）
        end: 块结束位置（字节）
        vocab_path: vocab文件路径
        merges_path: merges文件路径
        special_tokens: 特殊token列表
        temp_dir: 临时文件目录
        chunk_id: 块ID
    
    Returns:
        临时文件路径
    """
    try:
        # 加载BPE tokenizer（每个worker都需要加载）
        tokenizer = BPE.from_files(vocab_path, merges_path, special_tokens)
        
        # 读取块内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(start)
            # 计算需要读取的字符数（近似）
            # 由于是UTF-8，我们按字节读取然后解码
            chunk_bytes = end - start
            f.seek(start)
            content = f.read(chunk_bytes)
            # 如果没读到末尾，继续读到块结束
            if f.tell() < end:
                remaining = end - f.tell()
                content += f.read(remaining)
        
        # 编码
        tokens = tokenizer.encode(content)
        
        # 转换为numpy uint16数组并保存为npy格式
        tokens_array = np.array(tokens, dtype=np.uint16)
        temp_file = os.path.join(temp_dir, f"chunk_{chunk_id}.npy")
        np.save(temp_file, tokens_array)
        
        return temp_file
    except Exception as e:
        print(f"Error encoding chunk {chunk_id}: {e}")
        # 返回空结果
        temp_file = os.path.join(temp_dir, f"chunk_{chunk_id}.npy")
        np.save(temp_file, np.array([], dtype=np.uint16))
        return temp_file


def encode_chunk_binary(
    file_path: str,
    start: int,
    end: int,
    vocab_path: str,
    merges_path: str,
    special_tokens: List[str],
    temp_dir: str,
    chunk_id: int,
) -> str:
    """
    编码一个文件块（使用二进制模式读取，更准确）
    
    Args:
        file_path: 原始文件路径
        start: 块起始位置（字节）
        end: 块结束位置（字节）
        vocab_path: vocab文件路径
        merges_path: merges文件路径
        special_tokens: 特殊token列表
        temp_dir: 临时文件目录
        chunk_id: 块ID
    
    Returns:
        临时文件路径
    """
    try:
        # 加载BPE tokenizer（每个worker都需要加载）
        tokenizer = BPE.from_files(vocab_path, merges_path, special_tokens)
        
        # 读取块内容
        # 由于我们在special token处分割，边界应该是字符边界
        # 使用二进制模式读取，然后解码
        with open(file_path, 'rb') as f:
            f.seek(start)
            chunk_bytes = f.read(end - start)
        
        # 解码为文本（errors='ignore'会处理任何边界问题）
        try:
            content = chunk_bytes.decode('utf-8', errors='ignore')
        except Exception:
            # 如果解码失败，尝试其他编码
            content = chunk_bytes.decode('latin-1', errors='ignore')
        
        # 编码
        tokens = tokenizer.encode(content)
        
        # 转换为numpy uint16数组并保存为npy格式
        tokens_array = np.array(tokens, dtype=np.uint16)
        temp_file = os.path.join(temp_dir, f"chunk_{chunk_id}.npy")
        np.save(temp_file, tokens_array)
        
        return temp_file
    except Exception as e:
        print(f"Error encoding chunk {chunk_id}: {e}")
        import traceback
        traceback.print_exc()
        # 返回空结果
        temp_file = os.path.join(temp_dir, f"chunk_{chunk_id}.npy")
        np.save(temp_file, np.array([], dtype=np.uint16))
        return temp_file


def merge_two_files(file1: str, file2: str, output_file: str) -> None:
    """
    合并两个npy文件（使用内存映射，支持流式读取，适用于大文件）
    
    Args:
        file1: 第一个文件路径
        file2: 第二个文件路径
        output_file: 输出文件路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file1):
            raise FileNotFoundError(f"File not found: {file1}")
        if not os.path.exists(file2):
            raise FileNotFoundError(f"File not found: {file2}")
        
        # 使用内存映射加载两个文件（流式读取，不一次性加载到内存）
        tokens1 = np.load(file1, mmap_mode='r')
        tokens2 = np.load(file2, mmap_mode='r')
        
        # 合并（numpy的concatenate会自动处理内存映射的数组）
        merged_tokens = np.concatenate([tokens1, tokens2])
        
        # 保存到输出文件
        np.save(output_file, merged_tokens)
        
        # 验证输出文件已创建
        if not os.path.exists(output_file):
            raise RuntimeError(f"Failed to create output file: {output_file}")
        
        # 删除临时文件
        if os.path.exists(file1):
            os.remove(file1)
        if os.path.exists(file2):
            os.remove(file2)
    except Exception as e:
        import traceback
        print(f"Error merging files {file1} and {file2} -> {output_file}: {e}")
        traceback.print_exc()
        raise  # 重新抛出异常，让调用者知道合并失败

def merge_files_sequential(file_list: List[str], output_file: str, temp_dir: str = None, pbar: tqdm = None, chunk_size: int = 10 * 1024 * 1024) -> None:
    """
    一次性合并多个文件到最终输出（流式方式，避免一次性加载所有文件到内存）
    使用内存映射和分批处理，加速合并过程
    
    Args:
        file_list: 文件路径列表
        output_file: 最终输出文件路径
        temp_dir: 临时文件目录（如果为None，使用输出文件所在目录）
        pbar: 进度条对象
        chunk_size: 每次处理的元素数量（默认约10M个uint16，约20MB）
    """
    if len(file_list) == 0:
        return
    
    if len(file_list) == 1:
        # 只有一个文件，直接重命名
        if file_list[0] != output_file:
            shutil.move(file_list[0], output_file)
        return
    
    try:
        # 步骤1: 计算所有文件的总长度
        total_len = 0
        file_lengths = []
        for file_path in file_list:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            # 使用内存映射打开文件获取长度（不加载数据）
            data = np.load(file_path, mmap_mode='r')
            file_len = len(data)
            file_lengths.append((file_path, file_len))
            total_len += file_len
            del data  # 立即释放内存映射
        
        if total_len == 0:
            # 所有文件都为空，创建空输出文件
            np.save(output_file, np.array([], dtype=np.uint16))
            # 删除输入文件
            for file_path, _ in file_lengths:
                if os.path.exists(file_path):
                    os.remove(file_path)
            return
        
        # 步骤2: 创建输出文件的内存映射（写入模式）
        temp_output = output_file + '.tmp'
        output_mmap = np.memmap(temp_output, dtype=np.uint16, mode='w+', shape=(total_len,))
        
        # 步骤3: 按顺序将所有文件的内容复制到输出文件中（分批处理）
        offset = 0
        for file_path, file_len in file_lengths:
            # 使用内存映射打开文件（只读模式，不加载到内存）
            input_mmap = np.load(file_path, mmap_mode='r')
            
            # 分批复制数据
            for start in range(0, file_len, chunk_size):
                end = min(start + chunk_size, file_len)
                # 从内存映射中读取chunk（创建小的数组副本）
                chunk = np.array(input_mmap[start:end], copy=True)
                output_mmap[offset:offset + len(chunk)] = chunk
                offset += len(chunk)
                output_mmap.flush()  # 确保数据写入磁盘
            
            # 释放输入文件的内存映射
            del input_mmap
            
            # 更新进度条
            if pbar:
                pbar.update(1)
        
        # 步骤4: 关闭输出内存映射
        del output_mmap
        
        # 步骤5: 将临时文件转换为npy格式
        temp_data = np.memmap(temp_output, dtype=np.uint16, mode='r', shape=(total_len,))
        np.save(output_file, temp_data)
        del temp_data
        
        # 删除临时文件
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        # 验证输出文件已创建
        if not os.path.exists(output_file):
            raise RuntimeError(f"Failed to create output file: {output_file}")
        
        # 步骤6: 删除所有输入文件
        for file_path, _ in file_lengths:
            if os.path.exists(file_path):
                os.remove(file_path)
        
    except Exception as e:
        import traceback
        print(f"Error in sequential merge: {e}")
        traceback.print_exc()
        raise


def binary_merge_files(
    file_list: List[str],
    output_file: str,
    temp_dir: str,
    max_workers: int = 64,
    pbar: tqdm = None,
    level: int = 0,
    threshold_mb: float = 512.0,
) -> None:
    """
    使用二叉树方式合并文件，并行化加速
    当文件大小达到阈值（默认512MB）时，切换到串行合并模式
    
    Args:
        file_list: 文件路径列表
        output_file: 最终输出文件路径
        temp_dir: 临时文件目录
        max_workers: 最大worker数
        pbar: 进度条对象
        level: 当前合并层级（用于生成唯一文件名）
        threshold_mb: 切换到串行合并的阈值（MB）
    """
    if len(file_list) == 0:
        return
    
    if len(file_list) == 1:
        # 只有一个文件，直接重命名
        if file_list[0] != output_file:
            shutil.move(file_list[0], output_file)
        return
    
    # 检查文件大小，决定是否切换到串行合并
    threshold_bytes = threshold_mb * 1024 * 1024
    should_use_sequential = False
    
    # 检查是否有文件达到阈值
    for file_path in file_list:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size >= threshold_bytes:
                should_use_sequential = True
                print(f"检测到文件大小达到阈值 ({file_size / (1024*1024):.2f} MB >= {threshold_mb} MB)，切换到串行合并模式")
                break
    
    if should_use_sequential:
        # 使用串行合并
        merge_files_sequential(file_list, output_file, temp_dir, pbar)
        return
    
    # 继续并行合并
    # 创建新的一层文件列表
    next_level_files = []
    merge_tasks = []
    merge_counter = 0  # 用于生成唯一的文件名
    
    # 两两配对
    for i in range(0, len(file_list), 2):
        if i + 1 < len(file_list):
            # 有一对文件需要合并
            file1 = file_list[i]
            file2 = file_list[i + 1]
            # 使用层级和计数器生成唯一文件名
            output = os.path.join(temp_dir, f"merge_L{level}_{merge_counter}.npy")
            merge_tasks.append((file1, file2, output))
            next_level_files.append(output)
            merge_counter += 1
        else:
            # 奇数个文件，最后一个直接传递到下一层
            next_level_files.append(file_list[i])
    
    # 并行执行合并任务
    if merge_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(merge_two_files, f1, f2, out)
                for f1, f2, out in merge_tasks
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                    if pbar:
                        pbar.update(1)
                except Exception as e:
                    print(f"Error in merge task: {e}")
    
    # 递归合并下一层
    binary_merge_files(next_level_files, output_file, temp_dir, max_workers, pbar, level + 1, threshold_mb)


def encode_file(
    file_path: str,
    vocab_path: str,
    merges_path: str,
    output_path: str,
    special_tokens: List[str] = None,
    chunk_size_mb: float = 1.0,
    max_workers: int = 64,
) -> None:
    """
    编码大文件
    
    Args:
        file_path: 输入文件路径
        vocab_path: vocab文件路径
        merges_path: merges文件路径
        output_path: 输出文件路径（npy格式，包含token ID数组，uint16类型）
        special_tokens: 特殊token列表
        chunk_size_mb: 块大小（MB）
        max_workers: 最大worker数
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    
    # 创建临时目录（在output目录下）
    output_dir = os.path.dirname(output_path) or '.'
    temp_dir = os.path.join(output_dir, f"temp_bpe_encode_{os.getpid()}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"临时文件目录: {temp_dir}")
    
    try:
        # 步骤1: 分块
        print("步骤1: 分块...")
        split_token = special_tokens[0].encode('utf-8')
        with open(file_path, 'rb') as f:
            boundaries = find_chunk_boundaries_1mb(f, chunk_size_mb, split_token)
        
        num_chunks = len(boundaries) - 1
        print(f"文件分为 {num_chunks} 个块")
        
        # 步骤2: 并行编码
        print("步骤2: 并行编码...")
        chunk_tasks = []
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            chunk_tasks.append((
                file_path, start, end, vocab_path, merges_path,
                special_tokens, temp_dir, i
            ))
        
        temp_files = []
        with tqdm(total=num_chunks, desc="编码进度", unit="chunk") as pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(encode_chunk_binary, *task)
                    for task in chunk_tasks
                ]
                for future in as_completed(futures):
                    try:
                        temp_file = future.result()
                        temp_files.append(temp_file)
                        pbar.update(1)
                    except Exception as e:
                        print(f"编码任务出错: {e}")
        
        # 按chunk_id排序
        temp_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        print(f"编码完成，共生成 {len(temp_files)} 个临时文件")
        
        # 步骤3: 二叉树合并
        print("步骤3: 合并临时文件...")
        # 计算合并轮数
        num_merges = len(temp_files) - 1
        with tqdm(total=num_merges, desc="合并进度", unit="merge") as pbar:
            binary_merge_files(temp_files, output_path, temp_dir, max_workers, pbar)
        
        print(f"合并完成，结果保存到: {output_path}")
        
        # 验证输出文件
        final_tokens = np.load(output_path, mmap_mode='r')
        print(f"最终token数量: {len(final_tokens):,}")
        del final_tokens  # 释放内存映射
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"已清理临时目录: {temp_dir}")


def encode_directory(
    data_dir: str,
    vocab_path: str,
    merges_path: str,
    output_dir: str,
    special_tokens: List[str] = None,
    chunk_size_mb: float = 1.0,
    max_workers: int = 64,
) -> None:
    """
    编码目录下的所有文件
    
    Args:
        data_dir: 数据目录路径
        vocab_path: vocab文件路径
        merges_path: merges文件路径
        output_dir: 输出目录路径
        special_tokens: 特殊token列表
        chunk_size_mb: 块大小（MB）
        max_workers: 最大worker数
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有txt文件
    txt_files = list(data_path.glob("*.txt"))
    
    if not txt_files:
        print(f"在 {data_dir} 中未找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个文件需要编码")
    
    # 处理每个文件
    for txt_file in tqdm(txt_files, desc="处理文件", unit="file"):
        output_file = output_path / f"{txt_file.stem}_encoded.npy"
        print(f"\n处理文件: {txt_file.name}")
        encode_file(
            str(txt_file),
            vocab_path,
            merges_path,
            str(output_file),
            special_tokens,
            chunk_size_mb,
            max_workers,
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="编码大文件")
    parser.add_argument("--input", type=str, help="输入文件路径或目录")
    parser.add_argument("--data-dir", type=str, help="数据目录路径（处理目录下所有文件）")
    parser.add_argument("--vocab", type=str, required=True, help="vocab文件路径")
    parser.add_argument("--merges", type=str, required=True, help="merges文件路径")
    parser.add_argument("--output", type=str, help="输出文件路径（单文件模式）")
    parser.add_argument("--output-dir", type=str, help="输出目录路径（目录模式）")
    parser.add_argument("--chunk-size", type=float, default=1.0, help="块大小（MB）")
    parser.add_argument("--max-workers", type=int, default=64, help="最大worker数")
    parser.add_argument("--special-tokens", type=str, nargs="+", default=["<|endoftext|>"], help="特殊token列表")
    
    args = parser.parse_args()
    
    if args.data_dir:
        # 目录模式
        if not args.output_dir:
            args.output_dir = os.path.join(args.data_dir, "encoded")
        encode_directory(
            args.data_dir,
            args.vocab,
            args.merges,
            args.output_dir,
            args.special_tokens,
            args.chunk_size,
            args.max_workers,
        )
    elif args.input:
        # 单文件模式
        if not args.output:
            # 默认输出到同目录
            input_path = Path(args.input)
            args.output = str(input_path.parent / f"{input_path.stem}_encoded.npy")
        encode_file(
            args.input,
            args.vocab,
            args.merges,
            args.output,
            args.special_tokens,
            args.chunk_size,
            args.max_workers,
        )
    else:
        parser.error("必须指定 --input 或 --data-dir")
