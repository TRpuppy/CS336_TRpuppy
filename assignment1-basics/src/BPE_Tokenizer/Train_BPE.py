from src.BPE_Tokenizer.Pre_Tokenizer import parallel_pretokenize
from collections import defaultdict
import json
import os
import time
import pickle
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] = None):
    '''
    Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
    tokenizer. Your BPE training function should handle (at least) the following input parameters:
    
    input_path: str. Path to a text file with BPE tokenizer training data.
    vocab_size: int. A positive integer that defines the maximum final vocabulary size (including the
    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
    otherwise affect BPE training.

    Your BPE training function should return the resulting vocabulary and merges:
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation.
    '''
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    start_time = time.time()
    token_freq = parallel_pretokenize(input_path, max_workers=50, special_tokens=special_tokens)
    end_time = time.time()
    print(f"pre tokenize time cost :{end_time - start_time}")

    start_time = time.time()
    merges = []
    #create initial vocabulary
    vocab = {i : bytes([i]) for i in range(256)}
    len_vocab = 256
    for special_token in special_tokens:
        vocab[len_vocab] = bytes(special_token.encode('UTF-8'))
        len_vocab += 1
    
    #covert each token to a list of int.
    tokens = []
    freqs = []
    for token, freq in token_freq.items():
        freqs.append(freq)
        int_list = list(token.encode('utf-8'))
        tokens.append(int_list)
    
    appear_index = defaultdict(dict)
    #appear_index : a dictionary of dictionary, the first key is the index pair, the second key is the index of the token
    #the value is how many times the pair appears in the token.

    # 初始化 appear_times 字典，记录每个索引对出现的次数
    appear_times = defaultdict(int)
    for i in range(len(tokens)):
        for j in range(len(tokens[i]) - 1):
            appear_times[(tokens[i][j], tokens[i][j + 1])] += freqs[i]
            if i not in appear_index[(tokens[i][j], tokens[i][j + 1])]:
                appear_index[(tokens[i][j], tokens[i][j + 1])][i] = 0
            appear_index[(tokens[i][j], tokens[i][j + 1])][i] += 1



    #start training
    while (len_vocab < vocab_size):
        # 找到出现次数最多的索引对
        if not appear_times:
            break
        
        # 找到最大出现次数
        max_count = max(appear_times.values())
        # 找到所有出现次数等于 max_count 的索引对
        candidates = [pair for pair, count in appear_times.items() if count == max_count]
        # 如果有平局，选择字符串对较大的一个
        if len(candidates) > 1:
            max_pair = max(candidates, key=lambda pair: (vocab[pair[0]], vocab[pair[1]]))
        else:
            max_pair = candidates[0]
        
        # 记录要合并的索引对和出现次数
        merge_count = appear_times[max_pair]
        
        # 合并最频繁的索引对
        vocab[len_vocab] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
        
        # 更新 appear_times 字典（在更新 tokens 之前）
        a, b = max_pair[0], max_pair[1]
        
        # 1. 移除被合并的索引对 (a, b)
        del appear_times[max_pair]
        # 优化：使用 appear_index 直接获取包含 (a, b) 的所有 token 索引
        # 这样就不需要遍历所有 token，只处理真正包含该 pair 的 token
        affected_tokens = list(appear_index[max_pair].keys())
        del appear_index[max_pair]  # 移除 (a, b) 的索引记录
        
        # 2. 只更新包含 (a, b) 的 token（优化：从 O(n) 降低到 O(k)，k 是包含该 pair 的 token 数量）
        for i in affected_tokens:
            freq = freqs[i]
            token = tokens[i]
            
            # 找到所有 (a, b) 的位置
            j = 0
            while j < len(token) - 1:
                if (token[j], token[j + 1]) == (a, b):
                    # 这个位置会被合并为 len_vocab
                    # 需要移除的索引对：
                    # - (prev, a) 如果 j > 0
                    # - (a, b) 已经移除
                    # - (b, next) 如果 j + 1 < len(token) - 1
                    
                    # 需要添加的索引对：
                    # - (prev, len_vocab) 如果 j > 0
                    # - (len_vocab, next) 如果 j + 1 < len(token) - 1
                    
                    if j > 0:
                        prev = token[j - 1]
                        # 移除 (prev, a) 的统计和索引
                        old_pair = (prev, a)
                        if old_pair in appear_times:
                            appear_times[old_pair] -= freq
                            if appear_times[old_pair] <= 0:
                                del appear_times[old_pair]
                        # 更新 appear_index
                        if old_pair in appear_index and i in appear_index[old_pair]:
                            appear_index[old_pair][i] -= 1
                            if appear_index[old_pair][i] <= 0:
                                del appear_index[old_pair][i]
                            if len(appear_index[old_pair]) == 0:
                                del appear_index[old_pair]
                        
                        # 添加 (prev, len_vocab) 的统计和索引
                        appear_times[(prev, len_vocab)] += freq
                        if i not in appear_index[(prev, len_vocab)]:
                            appear_index[(prev, len_vocab)][i] = 0
                        appear_index[(prev, len_vocab)][i] += 1
                    
                    if j + 1 < len(token) - 1:
                        next_token = token[j + 2]
                        # 移除 (b, next) 的统计和索引
                        old_pair = (b, next_token)
                        if old_pair in appear_times:
                            appear_times[old_pair] -= freq
                            if appear_times[old_pair] <= 0:
                                del appear_times[old_pair]
                        # 更新 appear_index
                        if old_pair in appear_index and i in appear_index[old_pair]:
                            appear_index[old_pair][i] -= 1
                            if appear_index[old_pair][i] <= 0:
                                del appear_index[old_pair][i]
                            if len(appear_index[old_pair]) == 0:
                                del appear_index[old_pair]
                        
                        # 添加 (len_vocab, next) 的统计和索引
                        appear_times[(len_vocab, next_token)] += freq
                        if i not in appear_index[(len_vocab, next_token)]:
                            appear_index[(len_vocab, next_token)][i] = 0
                        appear_index[(len_vocab, next_token)][i] += 1
                    
                    j += 2  # 跳过已处理的 (a, b)
                else:
                    j += 1
        
        # 更新 tokens：将所有 (max_pair[0], max_pair[1]) 替换为 len_vocab
        # 优化：只更新受影响的 token，而不是遍历所有 token
        for i in affected_tokens:
            new_token_list = []
            j = 0
            while j < len(tokens[i]):
                if j < len(tokens[i]) - 1 and (tokens[i][j], tokens[i][j + 1]) == max_pair:
                    new_token_list.append(len_vocab)
                    j += 2
                else:
                    new_token_list.append(tokens[i][j])
                    j += 1
            tokens[i] = new_token_list
        
        len_vocab += 1
        if len_vocab % 200 == 0:
            print(f"training progress: {len_vocab}/{vocab_size}")
    end_time = time.time()
    print(f"train bpe time cost :{end_time - start_time}")
    return vocab,merges

def bytes_to_string(b: bytes) -> str:
    """
    将 bytes 转换为可读的字符串。
    尝试使用 UTF-8 解码，如果失败或包含控制字符则使用转义形式。
    """
    try:
        decoded = b.decode('utf-8')
        # 如果包含控制字符（除了常见的空白字符），使用转义形式
        if any(ord(c) < 32 and c not in '\n\r\t' for c in decoded):
            return repr(b)[2:-1]  # 去掉 b'...' 的 b' 和 '
        return decoded
    except (UnicodeDecodeError, UnicodeError):
        return repr(b)[2:-1]  # 去掉 b'...' 的 b' 和 '

def save_vocab_pickle(vocab: dict[int, bytes], output_path: str, special_tokens: list[str] = None):
    """
    将vocab以bytes形式存储到pickle文件。
    
    Args:
        vocab: dict[int, bytes] 词汇表，从token ID到bytes的映射
        output_path: str 输出pickle文件路径
        special_tokens: list[str] 特殊token列表（可选）
    """
    data = {
        'vocab': vocab,              # dict[int, bytes]
        'special_tokens': special_tokens  # list[str] | None
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ 已保存vocab到pickle文件: {output_path}")
    print(f"  Vocab大小: {len(vocab)}")

def save_merges_pickle(merges: list[tuple[bytes, bytes]], output_path: str):
    """
    将merges以bytes形式存储到pickle文件。
    
    Args:
        merges: list[tuple[bytes, bytes]] 合并规则列表
        output_path: str 输出pickle文件路径
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(merges, f)
    
    print(f"✓ 已保存merges到pickle文件: {output_path}")
    print(f"  Merges数量: {len(merges)}")

if __name__ == "__main__":
    file_path = "data/owt_train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(file_path, 32000, special_tokens=special_tokens)
    
    # 保存为pickle文件（分别保存vocab和merges）
    vocab_pickle_path = "output/vocab_owt_train.pkl"
    merges_pickle_path = "output/merges_owt_train.pkl"
    save_vocab_pickle(vocab, vocab_pickle_path, special_tokens=special_tokens)
    save_merges_pickle(merges, merges_pickle_path)