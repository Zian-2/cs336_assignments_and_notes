
本文为cs336的assignment1中分词器对应章节的参考解答和课程笔记。   

1 本文范围内的作业暂时还不需要用到GPU  
2 本课程并未提供好系统适配，所以事实上不推荐在win上完成作业。在你的服务器or虚拟机上完成可以避免很多麻烦。
# Setup：

## 环境配置：

按照https://github.com/stanford-cs336/assignment1-basics/tree/main 中README的说明配置并测试uv。  

如果你在Windows下，uv run pytest 时会出现问题，因为你没有办法import resource。(实际上，不推荐在windows上完成作业，后续还会出现其他问题……)  
尝试如下操作：  
打开\assignment1-basics\tests\test_tokenizer.py， 删除第5行的import resource，修改为：  

```python
try:
    import resource
except ImportError:
    resource = None
```

然后重新uv run pytest。  

## 数据下载：

下载 TinyStories data 和 subsample of OpenWebText：  
课程原生使用的是wget； windows下，推荐使用curl。试着使用如下命令：
```powershell
# 创建并进入数据目录
mkdir -p data
cd data

# 下载 TinyStories
curl.exe -L -o TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl.exe -L -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# 下载 OpenWebText
curl.exe -L -o owt_train.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
curl.exe -L -o owt_valid.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

# 解压 .gz 文件
# Windows 10/11 自带 tar.exe，可以替代 gunzip
tar.exe -xvzf owt_train.txt.gz
tar.exe -xvzf owt_valid.txt.gz

cd ..
```

如果解压不成功，考虑使用python原生解压。  
<!--more-->

先输入python进入交互模式，然后：
```python
import gzip
import shutil

with gzip.open('owt_valid.txt.gz', 'rb') as f_in:
    with open('owt_valid.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

```

<br> <br><br> 
# 1 作业总览
你将构建训练标准 Transformer LM 所需的所有组件，并训练一些模型。

### 你将实现：

1 BPE 分词器 (Byte-pair encoding tokenizer)：第 2 节   
2 Transformer 语言模型 (LM)：第 3 节  
3 交叉熵损失函数与 AdamW 优化器：第 4 节  
4 训练循环：支持模型和优化器状态的序列化与加载（保存与读取）：第 5 节  


### 你将完成如下任务：

1 在 TinyStories 数据集上训练一个 BPE 分词器。   
2 对数据集运行训练好的分词器，将其转换为整数 ID 序列。   
3 在 TinyStories 数据集上训练 Transformer 语言模型。   
4 使用训练好的模型生成样本并评估困惑度 (Perplexity)。  
5 在 OpenWebText 数据集上训练模型，并将达到的困惑度结果提交到排行榜。  
 

### 允许使用的工具：

课程希望你从0开始搭组件，所以不得使用`torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义，除了：

1 `torch.nn.Parameter`   
2 `torch.nn` 中的容器类（`Module`, `ModuleList`, `Sequential` 等）   
3 `torch.optim.Optimizer` 基类   


### 关于 AI 工具：

1 允许使用大语言模型进行低级的编程问题或提问高级的概念问题的咨询，但严禁直接使用 AI 来完成作业题目。
2 建议在完成作业时关闭 IDE 中的 AI 自动补全（如 Cursor Tab, GitHub CoPilot）。


### 数据集获取：

本次作业使用两个预处理过的数据集：TinyStories 和 OpenWebText。两者都是大型纯文本文件。
校内学生可在实验室机器的 `/data` 目录下找到。校外人员/自学者可根据**README.md中的命令下载。**


### 降低规模的建议：

后续课程会提供建议，解释如何在 CPU 或 MPS 环境下进行调整。



<br> <br> <br>
# 2 BPE(Byte-Pair Encoding)分词器

## 2.1 unicode标准

Unicode标准将字符映射到整数code points上。例如"s" 的code point 是115（或写作十六进制的U+0073），”牛”对应29275。

python中，有ord()和chr()函数：
```python
>>>ord('牛')
29275
>>> chr(29275)
'牛'
```


### Problem (unicode1): unicode介绍 （1分）
(a)`chr(0)` 返回的是什么字符？  
`chr(0)` 返回的是 Unicode 编码为 0 的字符，即空字符（Null Character）。

(b) 它的字符串表示形式 (`__repr__()`) 与打印形式有何不同？  
在 Python 中，其字符串表示形式（`__repr__()`）会显示为转义序列 `'\x00'`，而打印该字符（`print()`）时通常是不可见的，在某些终端里可能显示为空格。

(c)当该字符出现在文本中时会发生什么？  
在 Python 字符串内部它可以正常存在并拼接，但在将其打印到终端或与底层 C 语言编写的程序交互时，它可能会被当做文本结束符而导致后面的内容被截断，或者干脆显示为一个空白区域。

<br> <br>

## 2.2 unicode编码

主要介绍了UTF-8。编码和解码UTF-8的函数包括encode()和decode()，如下：
```python
>>> test_string = "hello! こんにちは!" 
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded) 
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!' 
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded) 
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character! 
>>> print(len(test_string)) 
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8")) 1
hello! こんにちは!
```

我们将整数的codepoints转换成了一个byte序列，它们会更易于处理。例如，因为任何文本实质上都被转化为介于 $0-255$ 之间的整数序列，只要词表包含了这 256 个基础字节，就不需要担心训练和搭建的过程中词表外（OOV）词汇。


### 课程文档外的补充：UTF-8字节序列对应规则

字节数1： 
对应Unicode编码范围：U+0000 至 U+007F 或 0~127 (7 bits) 
对应utf-8字节序列：0xxxxxxx

字节数2： 
对应Unicode编码范围：U+0080 至 U+07FF 或 128~2047 (11 bits) 
对应utf-8字节序列：110xxxxx 10xxxxxx

字节数3：
对应Unicode编码范围：U+0800 至 U+FFFF 或 2048~65535 (16 bits) 
对应utf-8字节序列：1110xxxx 10xxxxxx 10xxxxxx

字节数4：
对应Unicode编码范围：U+10000 至 U+10FFFF 或 65536~1114111 (21 bits) 
应utf-8字节序列：11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

对应方式：先把unicode编码翻译成二进制，在填入对应字节序列的'x'处。（长度不够填则在左侧补0）。



### Problem (unicode2): unicode编码（3分）

(a) 为什么 Tokenizer 训练更偏好 UTF-8而不是UTF-16, UTF-32？  
UTF-8 是变长编码，能够将常用的 ASCII 字符保持为单字节，避免了 UTF-16/32 在处理英文或代码时产生大量冗余零字节。

(b) 错误的解码函数分析：  
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes): 
	return "".join([bytes([b]).decode("utf-8") for b in bytestring]) 
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")) 'hello'
```

错误输入：如`b'\xe4\xb8\xad'` （汉字“中”的编码字节流）
因为UTF-8 中的中文字符或复杂符号是由 2 到 4 个字节共同组成的，单独解码其中任何一个字节都会因不符合 UTF-8 规范而报错（或产生乱码）。

(c)给出无法解码的字节：  
如`b'\xff\xff'`。  
原因：任何以 `11111xxx` 开头的字节都没有对应的 5 字节或更长的有效模板，无法解码为任何 Unicode 字符。

<br> <br>
## 2.3 子词分词方案（subword tokenization）

将词语拆成byte序列之后，仍然不能逐个byte拆分来看作为token：这会导致序列变得极长，训练的step变长，计算量也增大（如transformer计算复杂度与序列长度成正比）；同时更长的序列也导致信息密度被稀释，网络寻找token间的关系变得更困难。

而subword tokenization就是指代这样一个折中的方案。它选择用一个更大的词汇表(vocabulary)来trade-off更短的序列。比如说，如果‘the’经常出现，我们就可以给它单独分配一个条目(entry)，词汇表维度+1，但把3个token缩短成了1个。

为了做这样的工作，Sennrich et al. (2016) 提出使用byte-pair encoding (BPE; Gage, 1994)。作为一种subword tokenization，它简单地基于出现频率，将经常出现的byte pair合并(merge)成一个未被使用的索引。基于BPE的subword tokenizer被称为BPE tokenizer。

<br> <br>
## 2.4 训练BPE分词器

### 1 初始化 vocabulary
初始化的vocabulary是从byte到整数ID的映射。因此，vocabulary的大小为256。

### 2 预分词（Pre-tokenization)

理论上，有了词汇表我们就可以开始进行上述的合并（merge）工作了。然而，有两个关键的问题：
i 每次合并的时候，都需要从头到尾过一遍语料库。这在计算上是很昂贵的。  
ii 直接合并会导致出现一些新的token，它们只有标点符号的区别（比如”dog.“和”dog!“），它们会拥有完全不同的ID，即使它们在语义上是完全相同的。  

为了解决上述问题，我们进行对语料库的预分词(pre-tokenize)，先把语料库切成单词。这是如何省下计算成本的？举个例子，当我们已经计数了'text'的出现次数（比如说10次），当我们需要计数'te'的出现次数时，就可以直接+=10，从而避免了每次遇到同一个单词时重复的计算。

早期分词方案包括直接通过空格进行切分（s.split(""))，但显然这不能解决上述提到的问题2。

于是，我们将使用的时一个基于正则表达式(regex)的分词器 (used by GPT-2; Radford et al., 2019)，可以在github.com/openai/tiktoken/pull/234/files中找到。如下：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```
这里的五个部分分别处理：缩写后缀（'ll, 've, 't , ‘s等）；连续字母（单词）；连续数字；连续标点符号；连续纯空格。

例子如下：
```python
>>> # requires `regex` package 
>>> import regex as re 
>>>  re.findall(PAT, "some text that i'll pre-tokenize") 
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

实际使用中，用findall内存容易溢出，建议使用re.finditer这一迭代计算的函数。


### 3 计算BPE合并&处理特殊token

现在我们可以开始按照前述方法计算合并了。只需要注意两点：
i  不能跨越预分词边界合并。`['some', 'text']`，那么 `e`（来自 some）和 `t`（来自 text）永远不会被统计在一起。  
ii 多个byte pair出现频率并列第一时，选择字典序最大（Lexicographically greater）的那一对。("A", "B"), ("A", "C"), ("B", "ZZ"), 和 ("BA", "A")频率相同时，在 ASCII 中 "BA" > "B" > "A"），因此 `max` 会选出 `('BA', 'A')`。
iii 有一些特殊token不能和其他合并，比如<|endoftext|>。它不应该被分成几个零碎的token，因此我们会给它安排一个固定的tokenID。  

### 一个训练的具体例子：
例如我们现在有如下corpus：
```python
low low low low 
low lower lower widest widest widest 
newest newest newest newest newest newest
```
, 且vocabulary里已经放好了<|endoftext|>这一special token。

#### 1 初始化vocabulary：
一个special token和256个byte value。

#### 2 Pre-tokenization: 
为了简化，我们仅使用用空格分隔，最终得到频率表：{low: 5, lower: 2, widest: 3, newest: 6}。为容易处理，将它写成dict[tuple[bytes], int]的格式，如{(l,o,w): 5 …}。

#### 3 合并：
数出上述例子中byte pair的出现频率：
```python
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}
```
('es') 和 ('st')并列，我们就取字典序更大的('st')。
于是表格变为：{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}.

如是循环，进行6次merge，可以得到新产生的vocabulary：
```python
['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']
```

于是新的词汇表变为：
```python
[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]
```

在此语境下，‘newest’ 将被分词为[ne, west]。

<br> <br>
## 2.5 实验：训练BPE

我们接下来再TinyStories数据集上训练一个BPE。你可以在Section1里找到它的下载方式。在开始之前，推荐你先大概看一眼里面都是什么内容。

### 1 并行处理预分词

训练过程中，预分词会是一个主要的瓶颈。你可以使用内置库`multiprocessing`并行化你的代码。  
你可以逐字使用以下链接中的入门代码来获取分块边界，然后使用这些边界将工作分配到各个进程：  
[https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py)
阅读此代码，你会发现它实现的是确保每个分块边界都切割在special_token之前。

### 2 预分词前，剔除特殊token不进行处理

用"|" .join(special_tokens)来re.split以移除<|endoftext|>

### 3 优化合并

naive BPE training（逐个合并，每次合并从头遍历一次）计算量太大。考虑到每次合并改变的计数只有这个合并前后的计数，我们可以动态处理合并，即每次合并只将前后的计数-1。如合并‘text’中的‘ex’，只需要将te和xt的计数-1，而不需要合并完之后再从头数一遍。

### Downscaling 提示

1 cProfile和scalene等分析工具可以分析你的实现中的瓶颈，于是你可以专注于分析这些瓶颈。
2 与其直接现在TInyStories的training set上训练，你可以先将validation set作为‘调试数据(debug dataset)'训练。后者只有22K个文档，前者有2.12M个文档。




### Problem (train_bpe): BPE训练（15分）

1 编写你的tokenizer，然后在adapter.py中import，最后uv run pytest tests/test_train_bpe.py。
需要注意，如果出现类似” Extra items in the left set: b'\r\n\r'  “的报错，这是因为linux文本会被windows自动将\n更改为\r\n。这也很好解决，直接处理一步.replace(b"\r\n", b"\n")即可。    

笔者的个人实现见  
https://github.com/Zian-2/cs336_assignments_and_notes/blob/main/assignment1-basics/cs336_basics/tokenizer.py  ，    
本节答案对应截止在#--2.6 Encoding & Decoding--#上方的内容。
对其的说明参考https://zian-2.github.io/2026/01/21/tokenizer.py/  

2 按照文章的建议，You should use profiling tools like **`cProfile`** or **`scalene`** to identify the bottlenecks in your implementation.  个人的实现，在主函数内部添加如下代码：
```python
def train_bpe(input_path, vocab_size, special_tokens, num_chunks = 1000):

    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable() # 开始分析

  
##############################################
#  你的完整的分词进程
##############################################

    profiler.disable() # 结束分析
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(15) # 打印耗时最长的 15 个函数

    return tokenizer
```

添加分析之后运行时长增加是正常的。注意在表格中只需要关注tottime（函数内部的运行时长，不包括调用的外部函数）。


### Problem (train_bpe_tinystories)：TinyStories上的训练（2分）

(a) 在 TinyStories 数据集上训练一个字节级（byte-level）的 BPE 分词器，词表最大容量为 10,000。确保在词表中加入 `<|endoftext|>` 特殊标记。将生成的词表（vocabulary）和合并规则（merges）序列化保存到磁盘以便后续检查。训练耗时多少小时？消耗了多少内存？词表中最长的 Token 是什么？它合理吗？

资源要求： 耗时 ≤ 30 分钟（不使用 GPU），内存 ≤ 30GB RAM。
提示：你可以使用multiprocessing，可以使训练达到2分钟以内（这里应该指的是训练集TinyStoriesV2-GPT4-train.txt）

(b) 对你的代码进行性能分析（Profile）。分词器训练过程中哪个部分最耗时？

#### Answer：
你需要先把TinyStories接上你的分词器。记录内存消耗峰值，同时用cProfile分析，最后输出longest token，并将vocab和merges导入到你的磁盘里。参考实现如下（不包括profile）：
```python
import os
import json
from .tokenizer import train_bpe

def main():
    # 路径配置
    input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000 
    special_tokens = ["<|endoftext|>"] 

    print("开始训练")
    
    # 调用训练函数
    tokenizer = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_chunks=1000
    )

    print("\n训练完成")
    print(f"词表大小: {len(tokenizer.vocab)}")
    print(f"合并次数: {len(tokenizer.merges)}")

    output_dir = "run_bpe_train_on_tinystories_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    readable_vocab = {int(k): list(v) for k, v in tokenizer.vocab.items()}
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(readable_vocab, f, indent=4)

    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in tokenizer.merges:
            # 将 bytes 转换为逗号分隔的数字字符串
            s1 = ",".join(map(str, list(p1)))
            s2 = ",".join(map(str, list(p2)))
            f.write(f"{s1} {s2}\n")

if __name__ == "__main__":
    main()
```

#### Deliverable:  
(a)  笔者的实现参考：https://github.com/Zian-2/cs336_assignments_and_notes/blob/main/assignment1-basics/cs336_basics/tokenizer.py 。    
在TinyStoriesV2-GPT4-train.txt上训练在40秒以内，内存峰值0.8938 GB。  
设备条件：28GB内存，Intel Core Ultra 7 255HX CPU。  
longest token: ' accomplishment'。输出合理。

(b)Profile : 你可以以任何方式完成Profile。在TinyStoriesV2-GPT4-train.txt上的参考profile如下：
```PowerShell
==================================================
Profile (Top 20 Functions):
         15808949 function calls (15808073 primitive calls) in 40.449 seconds

   Ordered by: internal time
   List reduced from 457 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  463/235   20.998    0.045    6.500    0.028 {built-in method _winapi.WaitForMultipleObjects}
     9743    7.911    0.001   14.576    0.001 D:\cs336_assignments_and_notes\assignment1-basics\cs336_basics\tokenizer.py:169(merge_step)
    23282    2.899    0.000    2.951    0.000 {built-in method builtins.max}
   277780    2.271    0.000    3.413    0.000 D:\cs336_assignments_and_notes\assignment1-basics\cs336_basics\tokenizer.py:115(_replace_pair_and_pair_counts)
       80    1.486    0.019    1.486    0.019 {built-in method _pickle.loads}
     1002    1.238    0.001    2.147    0.002 D:\Program Files\Python313\Lib\collections\__init__.py:673(update)
  6434620    1.028    0.000    1.028    0.000 {method 'get' of 'dict' objects}
  2538958    0.486    0.000    0.486    0.000 {method 'add' of 'set' objects}
  1362761    0.374    0.000    0.374    0.000 {method 'discard' of 'set' objects}
  689/189    0.323    0.000   38.936    0.206 {built-in method _winapi.ReadFile}
        1    0.217    0.217   40.447   40.447 D:\cs336_assignments_and_notes\assignment1-basics\cs336_basics\tokenizer.py:218(train)
  3008052    0.190    0.000    0.190    0.000 {built-in method builtins.len}
   277795    0.156    0.000    0.156    0.000 {method 'pop' of 'dict' objects}
        1    0.135    0.135    0.196    0.196 D:\cs336_assignments_and_notes\assignment1-basics\cs336_basics\tokenizer.py:149(pair_count)
  1370918    0.133    0.000    0.133    0.000 {method 'append' of 'list' objects}
       20    0.123    0.006    0.123    0.006 {built-in method _winapi.CreateProcess}
       78    0.092    0.001    0.374    0.005 D:\Program Files\Python313\Lib\multiprocessing\connection.py:246(recv)
  794/768    0.051    0.000    0.128    0.000 {method 'GetOverlappedResult' of '_winapi.Overlapped' objects}
      157    0.050    0.000    0.050    0.000 {method 'write' of '_io.BytesIO' objects}
   270267    0.047    0.000    0.048    0.000 D:\cs336_assignments_and_notes\assignment1-basics\cs336_basics\tokenizer.py:178(<genexpr>)

==================================================
```


### Problem (train_bpe_expts_owt): OpenWebText上的训练（2分）

(a)在OpenWebText上训练，使用vocab_size = 32000。把你得到的vocab和merges存入硬盘。词表中longest token是什么？合理吗？  
资源要求： 耗时 ≤ 12小时（不使用 GPU），内存 ≤ 100GB RAM。  
(b)  对比你分别在TinyStories和OpenWebText上得到的分词器。

#### Answer:  
(a) uv run python -m cs336_basics.run_bpe_train_on_openwebtext。
longest token基本都是各种长横线或者下划线，你可以通过数据清洗试图避免这一点，但是owt里东西确实比较杂，笔者尝试了一些方法都不能清洗干净。  

(b) id小的词差不多，id大的词区别还是比较明显。owt的词明显更多简写，更不日常，更不口语化；Tiny就有些童话色彩。同样取10000附近的词举例：
```plaintext
OWT:
"9985": "82",
"9986": " signals",
"9987": " oxy",
"9988": " eager",
"9989": "igg",
"9990": "ERS",
"9991": " unprecedented",
"9992": " mood",
"9993": " custody",
"9994": " bankrupt",
"9995": " asylum",
"9996": " acknowledge",
"9997": "reek",
"9998": "endar",
"9999": "books"


TinyStories:
"9985": " Froggy", 
"9986": " wrapper", 
"9987": " Reddy", 
"9988": " Hops", 
"9989": " Crusty", 
"9990": " whiskers", 
"9991": " nicest", 
"9992": " improving", 
"9993": " booth", 
"9994": " Land", 
"9995": " Surrender", 
"9996": " Rocky", 
"9997": " meadows", 
"9998": " imaginary", 
"9999": " bold"
```

作为对比，再给出32000附近的OWT：  
```plaintext
"31985": " coated",
"31986": " bland",
"31987": " bending",
"31988": " bamboo",
"31989": " assurances",
"31990": " ambassadors",
"31991": " alum",
"31992": " Yee",
"31993": " Worse",
"31994": " Ware",
"31995": " Ves",
"31996": " TED",
"31997": " surveillance",
"31998": " Sequ",
"31999": " Schaefer"
```
<br> <br>
## 2.6 BPE：编码和解码

根据已给的词汇表和merges实现任意文本和vocabulary里的token IDs的互相转换（encode and decode) 。  

### 2.6.1 Encoding 文本
步骤：  
1 预分词 ：这一步和训练的时候是一样的。  
2 merge：拿出你的vocabulary和merges，按照merges创建时的顺序应用到预分词(pre-token)上。  

一个merge的具体例子：
假设输入是 `'the cat ate'`，  
词表： `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`   
合并 (merges) 是： `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`

首先预分词，拆成 `['the', ' cat', ' ate']`，然后对每个预分词应用合并。

第一个词 `'the'` 最初表示为 `[b't', b'h', b'e']`。查看合并列表，按顺序逐个应用可用的合并；第一个是 `(b't', b'h')`，应用它，`[b't', b'h', b'e']`转换为 `[b'th', b'e']`。下一个可用的合并是 `(b'th', b'e')`，将 `[b'th', b'e']`转换为 `[b'the']`。最后，再次查看合并列表，没有更多可应用的合并则结束。因此得到 'the' 对应的整数序列为 `[9]`。
同样地：`' cat'` 在应用合并后表示为 `[b' c', b'a', b't']`，变为整数序列 `[7, 1, 5]`。
 `' ate'` 合并后为 `[b' at', b'e']`，变为整数序列 `[10, 3]`。  
因此，编码输入字符串的最终结果是 `[9, 7, 1, 5, 10, 3]`。

3 special_tokens： 你的分词器应当能够恰当地对special tokens进行处理。  
4 内存方面的考量：文件文本常常不能直接塞进内存里，所以还是需要切成chunks再逐个处理，这样无论文件多大，所需的内存才是恒定的，而不是随着文件大小增长所需的内存线性增大。当然，需要保证切开的时候不能把一个token切开。  


### 2.6.2 Decoding 文本

Decoding，显然地，就是直接查找ID对应的token然后把它们连接起来，最后再把这些bytes转换成strings。  
然而，需要注意的是，不是所有输入的ID都能被转换成合法的Unicode strings，所以我们需要把这些畸形的(malformed)bytes替换成官方的unicode替换字符 `U+FFFD`（参考：`https://en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character`）。在bytes.decode函数中，参数errors控制了unicode decoding会怎样处理这些错误的bytes。比如说，用`errors = 'replace' `可以自动地把那些畸形的数据替换成替换标记。  


### Problem (tokenizer): 实现分词器（15分）
要求：实现一个 Tokenizer class，给它输出一个vocabulary和一个merges列表，可以把text文本encode成整数ID或把整数IDdecode成text文本。同时它还应该支持用户提供的特殊token（如果还不在词表里，就把他们添加到词表里。）课程推荐你的把接口写成如下形式：

`def __init__(self, vocab, merges, special_tokens=None)` 从给定的词表、合并列表和（可选的）特殊 Token 列表构造分词器。接受参数包括：  
`vocab: dict[int, bytes]`
`merges: list[tuple[bytes, bytes]]`
`special_tokens: list[str]  |  None = None`  

`def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)` 一个类方法。从序列化的vocab文件和merges文件（格式与 BPE 训练代码输出的格式相同）构造并返回一个 `Tokenizer` 实例。参数：   
`vocab_filepath: str`  
`merges_ffilepath:  str`  
`special_tokens:  list[str]  |  None = None`  

`def encode(self, text: str) -> list[int]`把输入的text文本转换成一列token IDs。  

`def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`  给定一个字符串迭代器（比如一个python文件句柄），返回一个有延迟的(lazily)token ID生成器。这是为了那些我们不能直接加载进内存的大型文件准备的。  

`def decode(self, ids: list[int]) -> str` 把token IDs解码成text文本。  

同样地，写好了之后先接入`adapters.get_tokenizer`，然后`uv run pytest tests/test_tokenizer.py`；你需要通过所有测试。  

#### Answer：
见  
https://github.com/Zian-2/cs336_assignments_and_notes/blob/main/assignment1-basics/cs336_basics/tokenizer.py  ，    
本节答案对应截止在# --2.6 Encoding & Decoding--#下方的内容。    

windows上测试依旧出现问题。最好还是在linux上跑吧……  
通过测试：
```python
============================= test session starts ==============================
platform linux -- Python 3.13.3, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/wangzian/Desktop/cs336_assignments_and_notes/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 25 items                                                             

tests/test_tokenizer.py::test_roundtrip_empty PASSED
tests/test_tokenizer.py::test_empty_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_character PASSED
tests/test_tokenizer.py::test_single_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_unicode_character PASSED
tests/test_tokenizer.py::test_single_unicode_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_ascii_string PASSED
tests/test_tokenizer.py::test_ascii_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string PASSED
tests/test_tokenizer.py::test_unicode_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens PASSED
tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken PASSED
tests/test_tokenizer.py::test_overlapping_special_tokens PASSED
tests/test_tokenizer.py::test_address_roundtrip PASSED
tests/test_tokenizer.py::test_address_matches_tiktoken PASSED
tests/test_tokenizer.py::test_german_roundtrip PASSED
tests/test_tokenizer.py::test_german_matches_tiktoken PASSED
tests/test_tokenizer.py::test_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_special_token_trailing_newlines PASSED
tests/test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_iterable_memory_usage PASSED
tests/test_tokenizer.py::test_encode_memory_usage XFAIL (Tokenizer.e...)

======================== 24 passed, 1 xfailed in 5.99s =========================
```

linux虚拟机下测试时间5.99s。只要逻辑不要太离谱（比如直接遍历merge）基本时间上不会有太大问题。


### 2.7 实验  

(a) 从 TinyStories 和 OpenWebText 中各随机抽取 10 份文档。使用你之前训练好的 TinyStories 分词器（词表大小 10K）和 OpenWebText 分词器（词表大小 32K），将这些抽样的文档编码为整数 ID。每个分词器的压缩比（字节/Token）是多少？

(b) 如果你用 TinyStories 的分词器去对 OpenWebText 的样本进行分词，会发生什么？比较其压缩比并（或）从定性角度描述发生的现象。

(c) 估算你的分词器的吞吐量（throughput)（例如：bytes/秒）。以此速度，分词整个Pile数据集（825GB 文本）需要多长时间？

(d) 使用你的 TinyStories 和 OpenWebText 分词器，将各自对应的训练集和开发集编码为整数 Token ID 序列。我们稍后将使用这些数据来训练语言模型。我们建议将这些 Token ID 序列序列化为 uint16 数据类型的 NumPy 数组。为什么选择 uint16 是一个合适的选择？

#### Answer：  
(a) Tiny 在 4.11左右， owt在4.42左右。可见owt三倍于Tiny的词表尺寸还是可以客观地提高压缩率。  
(b) 压缩率在3.3左右。可见tiny的词表相对于owt的内容还是有比较大局限。
(c) 在Tiny_valid上验证，吞吐量: 5.5023 MB/s。预估分词 Pile (825GB): 42.65 小时。
(d)  1 无符号这一点和tokenID是相同的   2 数值范围到65536，比较适配词表大小


--字数统计等