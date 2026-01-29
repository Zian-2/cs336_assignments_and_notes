```python
import os
from collections import Counter
import regex as re
import multiprocessing
from .pretokenization_example import find_chunk_boundaries
```

Counter : 一种特殊的数据类型，本质上是字典，下面你会看到它在计数层面上会有一些方便。
regex：引入正则表达式
multiprocessing: 用于并行处理

```python
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

```

五个部分分别处理：缩写后缀（'ll, 've, 't , ‘s等）；连续字母（单词）；连续数字；连续标点符号；连续纯空格。  
正则表达式的编译是比较昂贵的，放在外面从而不用每次都编译一遍。

pre_tokenization：输入分块了的chunk(需要先在下方process_chunk中转成str格式，不能是bytes流)； 输出分好词的列表，如['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']。

special_pat一行：用于切开从special_token，将"textA special_token textB" 切成"textA“，”special_token“，”textB"。

matches = PAT.finditer()的使用：matches是迭代器，需要用m.group来调用正则出来的单词。

```python
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
        
```

process_chunk： 1 输入文本中任意的start和end位置，2 为了避免windows自动将\n更改为\r\n的问题，替换\r\n  3 将其解码为str  4 调用前面的预分词(pre_tokenization0)切成单词  5 将单词编码回bytes流  6 对单词计数。7 返回一个包含了单词计数的Counter类型。注意这里的单词都是bytes形式。

```python
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
    
    
```

开始计数，并开始创建题目要求的vocab, merges。为此创建一个类并初始化参数。

```python
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
```

self.counting_init: 输入input_path, special_tokens, num_chunks（为了并行处理将原文本切成的大致块数），利用之前的process_chunk进行单词计数，然后合并所有结果。过程包括：

1 切割：按照课程提供的find_chunk_boundaries及其接口切割原文本。详情见https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py  。

2 并行：利用multiprocessing , 主要函数：  
i multiprocessing.cpu_count()，输出你的cpu的核数。    
ii  with multiprocessing.Pool(processes = num_cpu) as pool  :  创建并行池，processes = 并行个数。      
iii  results = pool.starmap(process_chunk, tasks)： 第一个形参是你要并行使用的函数， 第二个tasks是一个列表，每个元素是一个tuple, 内含你要传入前面那个函数的所有形参。

3 将Counter转回dict。

```python
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
```

 _ replace_pair_and_pair_counts：输入一个单词（bytes形式），它对应的计数，要合并的pair，以及给合并后的新词分配的new_id 。输出新的单词，并在这个过程中更新self.pair_counts。  
 容易理解，过程为先创建单词，再更新计数。注意此处更新应当只对有所更新的单词进行重新计数，否则pytest中的test_train_bpe_speed有1.5秒的速度要求将难以达成。

```python
    def pair_count(self, word_counts):
        # 第一次pair计数
        
        # 对pair计数（第一次）
        for word, count in word_counts.items():
            word_byte_pair = zip(word[:-1], word[1:])
            # 逐个地 += count
            for pair in word_byte_pair:
                self.pair_counts[pair] += count
        

```

我们已经有了对单词的计数，开始自然地对pair计数。   
1 注意zip()的语法：输入两个长度相同的tuple，两两组合。  
2 注意这里的计数方式符合我们的预分词目的。

```python
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


```

merge_step包含一步的step。输入：word_counts, new_id，返回new_word_counts，并在这个过程中更新self.vocab, self.merges 。  过程包括：  
1 找出频率最高的pair  
2 通过_replace_pair_and_pair_counts的更新，合并这些更新，产生新的单词计数new_word_counts传给下一步迭代。更新对pair的计数。  
3 更新self.vocab和self.merges。

```python    
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


```

train：合并上述过程。包括：
1 先进行第一步单词计数counting_init  
2 进行第一步pair计数pair_count  
3 迭代刚才的merge_step。  得到完整的vocab和merges
4 在最终的vocab开头给special_tokens分配固定的id。

```python
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
```

train_bpe: 在类外面。用于创建新实例，和adapter.py相接。
1 创建一个新的tokenizer实例，进行训练。  
2 按照课程的要求，使用cProfile进行分析。