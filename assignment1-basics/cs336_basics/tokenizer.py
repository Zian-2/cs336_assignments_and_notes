import os
from collections import Counter
import regex as re
import multiprocessing
from .pretokenization_example import find_chunk_boundaries


# 预编译正则表达式
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def pre_tokenization(chunk,special_tokens)->list[str]:
        # 预分词：先从special_token切开，再过一遍正则切成单词
        result = []
        # 切割special_tokens
        special_pat = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
        parts = re.split(special_pat, chunk)
        
        for part in parts:
            if not part:
                continue
            if part in special_tokens:
                continue

            # 预分词
            matches = PAT.finditer(part)
            for m in matches:
                result.append(f"{m.group()}")

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
        num_cpu = multiprocessing.cpu_count()
        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((input_path, start, end,special_tokens))
        
        with multiprocessing.Pool(processes = num_cpu) as pool:
            results = pool.starmap(process_chunk, tasks)

        # 转成dict[bytes:int]
        final_counts = Counter()
        for res in results:
            final_counts.update(res)

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

        # 动态更新pair计数
        if has_pair == True:
            for p in zip(word[:-1], word[1:]):
                self.pair_counts[p] -= count

            for p in zip(new_word[:-1], new_word[1:]):
                self.pair_counts[p] += count

        return tuple(new_word)



    def pair_count(self, word_counts):
        # 第一次pair计数
        
        # 对pair计数（第一次）
        for word, count in word_counts.items():
            word_byte_pair = zip(word[:-1], word[1:])
            # 逐个地 += count
            for pair in word_byte_pair:
                self.pair_counts[pair] += count
        


    def merge_step(self, word_counts, new_id):
        # merge迭代

        # 找出应合并的pair
        best_pair = max(
            self.pair_counts.items(), 
            key=lambda item: (
                item[1],                                       
                (self.vocab[item[0][0]], self.vocab[item[0][1]]) # 合并后的实际 bytes 内容
            )
        )[0]
        
        new_word_counts = {}
        # 合并pair并更新对单词和pair的计数
        for word, count in word_counts.items():
            if len(word) > 1 and (best_pair[0] in word and best_pair[1] in word):
                word = self._replace_pair_and_pair_counts(word, count, best_pair, new_id)
            new_word_counts[word] = new_word_counts.get(word, 0) + count

        self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]] 
        self.merges.append(( self.vocab[best_pair[0]], self.vocab[best_pair[1]] ))

        return new_word_counts


    
    def train(self, input_path, vocab_size, 
              special_tokens, num_chunks = 1000, new_id = None, 
              ):
        # 迭代 merge_step

        # 单词计数
        word_counts = self.counting_init(input_path, special_tokens, num_chunks = 1000)

        # 在最前面加上special_tokens 
        actual_merge_steps = vocab_size - 256 - len(special_tokens)

        # 数pair（第一次）
        self.pair_count(word_counts)

        # 迭代merge
        for i in range(actual_merge_steps):
            new_id = 256 + i
            word_counts = self.merge_step(word_counts, new_id)

        offset = len(special_tokens)
        new_vocab = {}
        
        for i, st in enumerate(special_tokens):
            new_vocab[i] = st.encode("utf-8", errors="ignore")
            
        for old_id, b in self.vocab.items():
            new_vocab[old_id + offset] = b
            
        self.vocab = new_vocab
        
        return self.vocab, self.merges



def train_bpe(input_path, vocab_size, special_tokens, num_chunks = 1000):

    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable() # 开始分析


    tokenizer = BPEtokenizer()
    tokenizer.train(input_path, vocab_size, 
              special_tokens, num_chunks = 1000, new_id = None, 
              )


    profiler.disable() # 结束分析
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(15) # 打印耗时最长的 15 个函数

    return tokenizer