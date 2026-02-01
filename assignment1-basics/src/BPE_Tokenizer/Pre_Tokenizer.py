import os
import concurrent.futures
import regex as re
from typing import List, Dict, BinaryIO
from collections import defaultdict
import multiprocessing
'''
Pretokenize a given file path. 
'''
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def merge_token_frequencies(freq_dicts: List[Dict[str, int]]) -> Dict[str, int]:
    """Merge multiple token frequency dictionaries

    Args:
        freq_dicts: List of token frequency dictionaries

    Returns:
        Merged global token frequency dictionary
    """
    global_freq = defaultdict(int)
    for freq_dict in freq_dicts:
        for token, count in freq_dict.items():
            global_freq[token] += count
    return dict(global_freq)


def pretokenize_chunk_streaming(file_path: str, start: int, end: int, special_tokens: List[str] = None, chunk_id: int = None) -> Dict[str, int]:
    """
    流式处理文件的一个chunk，避免一次性加载到内存。
    Args:
        file_path: 文件路径
        start: chunk的起始位置（字节）
        end: chunk的结束位置（字节）
        special_tokens: List of special tokens to split on
        chunk_id: chunk的ID，用于进度输出
    Returns:
        A dictionary of token frequencies, where the key is the token and the value is the number of occurrences
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    try:
        token_freq = defaultdict(int)
        split_token = special_tokens[0]
        split_token_bytes = split_token.encode('utf-8')
        
        buffer = b''
        processed_bytes = 0  # 已处理的字节数
        progress_threshold = 1 * 1024 * 1024  # 每100MB输出一次进度
        last_reported = 0  # 上次报告的字节数
        
        worker_id = multiprocessing.current_process().pid
        chunk_info = f"Chunk {chunk_id}" if chunk_id is not None else f"Worker {worker_id}"
        chunk_total = end - start
        
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining_bytes = end - start
            
            # 流式读取，每次读取一小块
            chunk_size = 4096  # Read ahead by 4k bytes at a time
            while remaining_bytes > 0:
                read_size = min(chunk_size, remaining_bytes)
                chunk_bytes = f.read(read_size)
                if not chunk_bytes:
                    break
                
                remaining_bytes -= len(chunk_bytes)
                buffer += chunk_bytes
                
                # 更新已处理的字节数（基于已读取的数据）
                processed_bytes = chunk_total - remaining_bytes
                
                # 每处理100MB输出一次进度
                if processed_bytes - last_reported >= progress_threshold:
                    processed_mb = processed_bytes / (1024 * 1024)
                    total_mb = chunk_total / (1024 * 1024)
                    progress_pct = (processed_bytes / chunk_total * 100) if chunk_total > 0 else 0
                    print(f"[{chunk_info}] 已处理: {processed_mb:.2f} MB / {total_mb:.2f} MB ({progress_pct:.1f}%)")
                    last_reported = processed_bytes
                
                # 处理buffer中完整的文档
                while True:
                    # 查找special token的位置
                    token_pos = buffer.find(split_token_bytes)
                    
                    if token_pos == -1:
                        # 没有找到special token，检查是否还有更多数据要读
                        if remaining_bytes == 0:
                            # 文件结束，处理剩余的buffer
                            if buffer:
                                try:
                                    text = buffer.decode('utf-8', errors='ignore')
                                    # 规范化换行符：将\r\n转换为\n，单独的\r也转换为\n
                                    text = text.replace('\r\n', '\n').replace('\r', '\n')
                                    if text:
                                        tokens = re.findall(PAT, text)
                                        for token in tokens:
                                            if not any(special_token in token for special_token in special_tokens):
                                                token_freq[token] += 1
                                except Exception as e:
                                    print(f"Error processing final buffer: {e}")
                            break
                        else:
                            # 还有更多数据，继续读取
                            break
                    else:
                        # 找到了special token，处理之前的文档
                        doc_bytes = buffer[:token_pos]
                        buffer = buffer[token_pos + len(split_token_bytes):]
                        
                        if doc_bytes:
                            try:
                                text = doc_bytes.decode('utf-8', errors='ignore')
                                # 规范化换行符：将\r\n转换为\n，单独的\r也转换为\n
                                text = text.replace('\r\n', '\n').replace('\r', '\n')
                                if text:
                                    tokens = re.findall(PAT, text)
                                    for token in tokens:
                                        if not any(special_token in token for special_token in special_tokens):
                                            token_freq[token] += 1
                            except Exception as e:
                                print(f"Error processing document: {e}")

        # 输出最终完成信息
        processed_mb = chunk_total / (1024 * 1024)
        print(f"[{chunk_info}] 完成! 总计处理: {processed_mb:.2f} MB, 发现 {len(token_freq)} 个唯一token")
        
        return dict(token_freq)
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return {}


def parallel_pretokenize(file_path: str, max_workers: int = None, special_tokens: List[str] = None) -> Dict[str, int]:
    """Parallelize pre-tokenization and return the frequency statistics of each token
    使用流式处理，避免一次性加载整个文件到内存

    Args:
        file_path: The path to the data file
        max_workers: The maximum number of processes, None means using the number of CPU cores
        special_tokens: List of special tokens to split on, defaults to ["<|endoftext|>"]

    Returns:
        A dictionary of token frequencies, where the key is the token and the value is the number of occurrences
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    # 使用字节级别的分块方法
    split_special_token = special_tokens[0].encode('utf-8')
    
    freq_dicts = []
    file_size = 0
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        boundaries = find_chunk_boundaries(f, max_workers, split_special_token)
    
    file_size_mb = file_size / (1024 * 1024)
    print(f"开始并行预分词: 文件大小 {file_size_mb:.2f} MB, 分为 {len(boundaries)-1} 个chunk, 使用 {max_workers} 个worker")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有块的处理任务，使用流式处理函数
        futures = []
        chunk_id = 0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # 传递文件路径和边界位置，让每个worker自己打开文件进行流式处理
            future = executor.submit(pretokenize_chunk_streaming, file_path, start, end, special_tokens, chunk_id)
            futures.append(future)
            chunk_id += 1

        # 收集结果
        completed_chunks = 0
        total_chunks = len(futures)
        for future in concurrent.futures.as_completed(futures):
            try:
                freq_dict = future.result()
                freq_dicts.append(freq_dict)
                completed_chunks += 1
                print(f"[总体进度] 已完成 {completed_chunks}/{total_chunks} 个chunk ({completed_chunks/total_chunks*100:.1f}%)")
            except Exception as exc:
                print(f'Chunk processing generated an exception: {exc}')
                freq_dicts.append({})  # Add an empty dictionary to avoid index errors
                completed_chunks += 1

    # Merge all frequency dictionaries
    print("正在合并所有chunk的词频统计...")
    global_freq = merge_token_frequencies(freq_dicts)
    print(f"预分词完成! 总共发现 {len(global_freq)} 个唯一token, 总出现次数: {sum(global_freq.values()):,}")
    return global_freq


# 测试代码
if __name__ == "__main__":
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    token_freq = parallel_pretokenize(file_path, max_workers=4)

    print(f"Total unique tokens: {len(token_freq)}")
    print(f"Total token occurrences: {sum(token_freq.values())}")

    # 显示最常见的10个tokens
    top_10 = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 most frequent tokens:")
    for token, freq in top_10:
        print(f"'{token}': {freq}")

    # 验证special tokens已被排除
    special_tokens = ["<|endoftext|>"]
    has_special = any(token in special_tokens for token in token_freq.keys())
    print(f"\nContains special tokens: {has_special}")

    # 示例：使用自定义special_tokens
    print("\n" + "="*50)
    print("Example with custom special_tokens:")
    custom_special_tokens = ["<|endoftext|>", "<|startoftext|>"]
    # 注意：这个示例假设文件中没有<|startoftext|>，只是为了演示
    token_freq_custom = parallel_pretokenize(file_path, max_workers=4, special_tokens=custom_special_tokens)
    print(f"With custom special_tokens: {len(token_freq_custom)} unique tokens")
