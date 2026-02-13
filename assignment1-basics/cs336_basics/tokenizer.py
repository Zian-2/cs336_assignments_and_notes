import os
from collections import Counter
from collections import defaultdict
import regex as re
import multiprocessing
from pretokenization_example import find_chunk_boundaries

import time


# 预编译正则
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def pre_tokenization(chunk, special_tokens)->list[str]:
    if not special_tokens:
        # 如果没有特殊 token，直接正则切词，不要用 re.split
        return [m.group() for m in PAT.finditer(chunk)]
    
    sorted_special = sorted(special_tokens, key=len, reverse=True)
    special_pat = "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
    parts = re.split(special_pat, chunk)
    
    result = []
    for part in parts:
        if not part: continue
        if part in special_tokens:
            result.append(part)
        else:
            matches = PAT.finditer(part)
            for m in matches:
                result.append(m.group())
    return result



def process_chunk(input_path, 
                    start,
                    end,
                    special_tokens)-> dict[tuple[int, ...], int]:
        # 每个核的并行任务：(预分词)，encode，然后计数

        with open(input_path, "rb") as f:
            f.seek(start)

            raw_data = f.read(end - start).replace(b"\r\n", b"\n")

            chunk = raw_data.decode("utf-8", errors="ignore")
            
            # 预分词
            chunk_split = pre_tokenization(chunk, special_tokens)

            # encode每个单词并计数
            word_counts = Counter()
            for word in chunk_split:
                word_b = tuple(word.encode("utf-8", errors="ignore"))
                word_counts[word_b] += 1

            return dict(word_counts) 
        


class BPEtokenizer:
    '''
    输入input_path(包含training data), vocab_size(int), special_tokens(list[str]).

    返回vocab(dict[int, bytes], int为原ID, bytes为token bytes), 

    和merges(list[tuple[bytes, bytes]],记录所有被合并的bytes). 
    '''
    def __init__(self, vocab=None, merges=None, special_tokens=None, pair_counts = Counter()):
        self.vocab = vocab if vocab is not None else {i: bytes([i]) for i in range(256)}
        self.merges = merges if merges is not None else []
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.pair_counts = Counter()
        self.pair_to_words = defaultdict(set)
        self.changed_pairs = set()
        self.candidates = set()
        self.current_max_freq = -1

        
    

    def counting_init(self, input_path, 
              special_tokens, num_chunks = 1000
              ) -> dict[tuple[int, ...], int]:
        # 初始化：第一次计数

        # 按边界切割
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, 
                desired_num_chunks = num_chunks, 
                split_special_token = special_tokens[0].encode("utf-8")
            )        

        # 并行处理
        print("start multiprocessing")
        num_cpu = multiprocessing.cpu_count()
        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((input_path, start, end, special_tokens))
        
        with multiprocessing.Pool(processes = num_cpu) as pool:
            results = pool.starmap(process_chunk, tasks)

        # 转成dict[bytes:int]
        final_counts = Counter()
        for res in results:
            final_counts.update(res)
        
        print("finish multiprocessing")
        return dict(final_counts)
    


    def _replace_pair_and_pair_counts(self, word, count, pair, new_id):
        
        # 更新单词
        new_word = []
        i = 0
        has_pair = False
        while i < len(word):
            if i < len(word)-1 and word[i] == pair[0] and word[i+1] == pair[1]:
                has_pair = True
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)

        # 动态更新pair计数
        if has_pair == True:
            for p in zip(word[:-1], word[1:]):
                self.pair_counts[p] -= count
                if self.pair_counts[p] <= 0:
                    del self.pair_counts[p]  # 加快merge_step和max
                self.pair_to_words[p].discard(word) 

            for p in zip(new_word[:-1], new_word[1:]):
                self.pair_counts[p] += count
                self.pair_to_words[p].add(new_word)

                self.changed_pairs.add(p)

        return new_word



    def pair_count(self, word_counts):
        # 第一次pair计数
        
        # 对pair计数（第一次）
        for word, count in word_counts.items():
            word_byte_pair = zip(word[:-1], word[1:])
            # 逐个地 += count
            for pair in word_byte_pair:
                self.pair_counts[pair] += count
        
        # 初始化倒排索引pair_to_words，避免后续重新查找
        for word in word_counts:
            for p in zip(word[:-1], word[1:]):
                self.pair_to_words[p].add(word)

        # 第一次生成self.candidates
        max_freq = max(self.pair_counts.values())
        self.candidates = {p for p, count in self.pair_counts.items() if count == max_freq}


    def merge_step(self, word_counts, new_id):
        # merge迭代

        # 找出应合并的pair

        # 更新candidates
        self.candidates.update(self.changed_pairs)
        self.changed_pairs = set()

        local_max = max((self.pair_counts[p] for p in self.candidates), default=0)

        if local_max < self.current_max_freq:
            self.current_max_freq = max(self.pair_counts.values())
            self.candidates = {p for p, count in self.pair_counts.items() if count == self.current_max_freq}
        
        else: 
            self.current_max_freq = local_max
            self.candidates = {p for p in self.candidates if self.pair_counts[p] == self.current_max_freq}

        # 对找出best_pair的优化
        if len(self.candidates) == 1:
            best_pair = next(iter(self.candidates))
        else:
            v = self.vocab
            best_pair = max(
                self.candidates,
                key=lambda p: (v[p[0]], v[p[1]])
            )
        


        # 对合并pair，词语更新，计数pair的优化, 速度较快
        affected_words = list(self.pair_to_words.get(best_pair, []))

        for old_word in affected_words:
            count = word_counts.pop(old_word)
            new_word = self._replace_pair_and_pair_counts(old_word, count, best_pair, new_id)

            word_counts[new_word] = word_counts.get(new_word, 0) + count

        self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]] 
        self.merges.append(( self.vocab[best_pair[0]], self.vocab[best_pair[1]] ))

        return word_counts




    
    def train(self, input_path, vocab_size, 
              special_tokens, num_chunks = 1000, new_id = None, 
              ):
        # 迭代 merge_step

        # 单词计数
        print(f"start word counting")
        word_counts = self.counting_init(input_path, special_tokens, num_chunks = 1000)
        print(f"finish word counting")

        # 在最前面加上special_tokens 
        actual_merge_steps = vocab_size - 256 - len(special_tokens)

        # 数pair（第一次）
        print(f"start merging")
        self.pair_count(word_counts)

        # 迭代merge
        for i in range(actual_merge_steps):
            new_id = 256 + i
            word_counts = self.merge_step(word_counts, new_id)
            if i % 10 == 0:
                print(f"iteration {i}")

        offset = len(special_tokens)
        new_vocab = {}
        
        for i, st in enumerate(special_tokens):
            new_vocab[i] = st.encode("utf-8", errors="ignore")
            
        for old_id, b in self.vocab.items():
            new_vocab[old_id + offset] = b
            
        self.vocab = new_vocab
        
        return self.vocab, self.merges



def train_bpe(input_path, vocab_size, special_tokens, num_chunks = 1000):

    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable() # 开始分析


    tokenizer = BPEtokenizer()
    tokenizer.train(input_path, vocab_size, 
              special_tokens, num_chunks = 1000, new_id = None, 
              )


    # profiler.disable() # 结束分析
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats(15) # 打印耗时最长的 15 个函数

    return tokenizer




# --------------------2.6 Encoding & Decoding-----------------------# 
import json

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy() # 后续可能把special_tokens加入
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}  # 查找表，否则直接merge遍历单词太慢


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        with open(vocab_filepath, "r", encoding = 'utf-8')as f:
            raw_vocab = json.load(f)
            # 转成 int: bytes
            vocab = {int(k): v.encode('utf-8') if isinstance(v, str) else bytes([v]) for k, v in raw_vocab.items()}
            
        merges = []
        with open(merges_filepath, "r", encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                if len(parts) == 2:
                    p0 = bytes(map(int, parts[0].split(',')))
                    p1 = bytes(map(int, parts[1].split(',')))
                    merges.append((p0, p1)) 
        
        return cls(vocab, merges, special_tokens)



    def decode(self, ids):
        # 逻辑简单: 翻译，串起来，decode
        byte_data = b"".join(self.vocab[id] for id in ids)
        return byte_data.decode('utf-8', errors = 'replace')

    def encode(self, text):
        if not text:
            return []
        
        # 预分词
        words = pre_tokenization(text, self.special_tokens)
        

        final_ids = []

        for word in words:
            # 整个单词就是一个Token，则直接查
            word_bytes = word.encode('utf-8')
            if word_bytes in self.byte_to_id:
                final_ids.append(self.byte_to_id[word_bytes])
                continue
            
            # 拆成字节
            tokens = [bytes([b]) for b in word_bytes]
            

            while len(tokens) > 1:
                # 找出rank 最小的
                pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]

                best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
                
                if best_pair not in self.bpe_ranks:
                    break

                new_tokens = []
                i = 0
                p0, p1 = best_pair
                target = p0 + p1
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i+1] == p1:
                        new_tokens.append(target)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            # 将最终合并出的 tokens 转为 ID
            for t in tokens:
                final_ids.append(self.byte_to_id[t])

        return final_ids


    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)