import pickle
from typing import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class LRUCache:
    """简单的LRU（Least Recently Used）缓存实现"""
    def __init__(self, max_size: int = 20000):
        """
        Args:
            max_size: 缓存的最大容量
        """
        self.max_size = max_size
        self.cache: OrderedDict[bytes, list[int]] = OrderedDict()
    
    def get(self, key: bytes) -> list[int] | None:
        """获取缓存值，如果存在则将其移到末尾（标记为最近使用）"""
        if key in self.cache:
            # 移动到末尾，表示最近使用
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: bytes, value: list[int]) -> None:
        """添加或更新缓存值"""
        if key in self.cache:
            # 如果已存在，更新值并移到末尾
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # 如果缓存已满，删除最久未使用的项（第一个）
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # last=False 表示删除第一个（最旧的）
            # 添加新项到末尾
            self.cache[key] = value
    
    def __contains__(self, key: bytes) -> bool:
        """支持 'key in cache' 语法"""
        return key in self.cache
    
    def __len__(self) -> int:
        """返回缓存大小"""
        return len(self.cache)
class BPE:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]|None = None):
        '''
        Construct a tokenizer from a given vocabulary, list of merges 
        and (optionally) a list of special tokens.
        '''
        self.vocab = vocab
        self.merges = merges

        #byte2token: dict[bytes, int], the index of the token in the vocabulary. It is the inverse of vocab.
        self.byte2token = {vocab[token]: token for token in vocab}
        #merge_index: list[tuple[int, int]], the index of the two tokens in the vocabulary that are merged.
        self.merge_index = [(self.byte2token[merge[0]], self.byte2token[merge[1]]) for merge in merges]
        
        # 预计算合并映射表：将 (token_id1, token_id2) 映射到合并后的 token_id
        # 这样可以避免在 apply_merges 中重复计算 merged_bytes 和 merged_token_id
        self.merge_map: dict[tuple[int, int], int] = {}
        # 预计算合并优先级：将 (token_id1, token_id2) 映射到它在 merge_index 中的位置
        # 位置越小，优先级越高（越早出现的 merge 规则优先级越高）
        self.merge_priority: dict[tuple[int, int], int] = {}
        for idx, merge_pair in enumerate(self.merge_index):
            if merge_pair not in self.merge_priority:
                self.merge_priority[merge_pair] = idx
            token_id1, token_id2 = merge_pair
            merged_bytes = self.vocab[token_id1] + self.vocab[token_id2]
            merged_token_id = self.byte2token[merged_bytes]
            self.merge_map[(token_id1, token_id2)] = merged_token_id

        self.special_tokens = special_tokens
        if special_tokens is None:
            self.special_tokens = ["<|endoftext|>"]

        # 使用LRU缓存来缓存merge结果
        self._merge_cache = LRUCache(max_size=20000)
        
        # 构建特殊token的快速查找结构：按首字符索引，按长度降序排序
        # 这样可以快速定位可能匹配的token，避免遍历所有token
        self._special_token_index: dict[str, list[tuple[str, int]]] = {}
        for token in self.special_tokens:
            if not token:
                continue
            first_char = token[0]
            if first_char not in self._special_token_index:
                self._special_token_index[first_char] = []
            self._special_token_index[first_char].append((token, len(token)))
        
        # 对每个首字符的token列表按长度降序排序（贪心匹配：优先匹配长的）
        for first_char in self._special_token_index:
            self._special_token_index[first_char].sort(key=lambda x: x[1], reverse=True)
    @classmethod
    def from_files(cls, vocab_pickle_path: str, merges_pickle_path: str, special_tokens=None):
        """
        Load vocab and merges data from pickle files.
        
        Args:
            vocab_pickle_path: vocab pickle file path
            merges_pickle_path: merges pickle file path

        Returns:
            BPE object
        """
        # Load vocab
        with open(vocab_pickle_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        vocab = vocab_data['vocab']                    # dict[int, bytes]

        # Load merges
        with open(merges_pickle_path, 'rb') as f:
            merges = pickle.load(f)                    # list[tuple[bytes, bytes]]
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        #1. pretokenize the text and get all tokens. Maybe need to modify Pre_Tokenizer.py to do it.
        # the pretokenize function should accept a string or a Iterable[str] and return a list of tokens.
        #2. apply all merge rules on each token. The process is similar to BPE training, but it can be done in parallel.
        # we keep a dictionary: list[bytes] -> list[int] to store initial token to merged token mapping.
        #3. return the list of tokens.
        
        # 如果没有特殊token，直接处理整个文本（同时规范化换行符）
        if not self._special_token_index:
            # 规范化换行符并处理
            normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
            tokens = re.finditer(PAT, normalized_text)
            result = []
            for token in tokens:
                result.extend(self.apply_merges(token.group().encode('utf-8')))
            return result
        
        # 单次遍历，边遍历边处理
        result = []
        i = 0
        text_len = len(text)
        normal_text_start = 0  # 当前普通文本段的起始位置（在原始text中的位置）
        
        def process_normal_text(start: int, end: int):
            """处理普通文本段：规范化换行符并应用BPE"""
            if end <= start:
                return
            
            # 规范化换行符
            normal_text = text[start:end]
            normal_text = normal_text.replace('\r\n', '\n').replace('\r', '\n')
            
            # 应用正则分词和BPE merge
            tokens = re.finditer(PAT, normal_text)
            for token in tokens:
                result.extend(self.apply_merges(token.group().encode('utf-8')))
        
        while i < text_len:
            # 尝试匹配特殊token
            matched = False
            char = text[i]
            
            # 使用索引快速查找可能匹配的特殊token
            if char in self._special_token_index:
                # 检查所有以当前字符开头的特殊token（已按长度降序排序）
                for special_token, token_len in self._special_token_index[char]:
                    # 检查是否有足够长度且匹配
                    # 只创建必要长度的切片（特殊token的长度，通常很短）
                    if i + token_len <= text_len and text[i:i+token_len] == special_token:
                        # 先处理之前累积的普通文本
                        if i > normal_text_start:
                            process_normal_text(normal_text_start, i)
                        
                        # 添加特殊token的token ID
                        result.append(self.byte2token[special_token.encode('utf-8')])
                        
                        # 更新位置
                        i += token_len
                        normal_text_start = i
                        matched = True
                        break
            
            if not matched:
                i += 1
        
        # 处理最后剩余的普通文本
        if normal_text_start < text_len:
            process_normal_text(normal_text_start, text_len)
        
        return result

    def apply_merges(self, token: bytes) -> list[int]:
        # 尝试从LRU缓存中获取
        cached_result = self._merge_cache.get(token)
        if cached_result is not None:
            return cached_result
        
        # 将 token bytes 转换为初始 token_id 列表（每个 byte 对应一个 token_id）
        word: list[int] = [self.byte2token[bytes([b])] for b in token]
        
        def getpair(word):
            pairs = []
            for i in range(len(word) - 1):
                pairs.append((word[i], word[i + 1]))
            return pairs
        
        pairs = getpair(word)
        if not pairs:
            result = word
            self._merge_cache.put(token, result)
            return result
        
        while True:
            # 找到优先级最高的 merge_pair（priority 越小优先级越高）
            merge_pair = min(pairs, key=lambda pair: self.merge_priority.get(pair, float("inf")))
            if merge_pair not in self.merge_priority:
                break
            
            # 在整个 word 序列中应用这个 rule（从左到右，不重叠）
            new_word = []
            merged_token_id = self.merge_map[merge_pair]
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == merge_pair:
                    new_word.append(merged_token_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
            if len(word) == 1:
                break
            pairs = getpair(word)
        
        result = word
        # 将结果存入LRU缓存
        self._merge_cache.put(token, result)
        return result

    def encode_iterable(self, iterable: Iterable[str], batch_size: int = 1000, max_workers: int = None) -> Iterator[int]:
        from concurrent.futures import ThreadPoolExecutor
        import os
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        # 批量收集文本
        batch = []
        for text in iterable:
            batch.append(text)
            if len(batch) >= batch_size:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for result in executor.map(self.encode, batch):
                        yield from result
                batch = []
        
        # 处理剩余的文本
        if batch:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(self.encode, batch):
                    yield from result

    def decode(self, tokens: list[int]) -> str:
        # 先将所有 token ID 转换成对应的 bytes
        bytes_list = [self.vocab[token] for token in tokens]
        # 将所有 bytes 连接起来
        combined_bytes = b''.join(bytes_list)
        # 最后对整个连接的 bytes 进行一次 utf-8 decode
        result = combined_bytes.decode('utf-8', errors='replace')
        return result