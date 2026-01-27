

# Setup

## 环境配置：

按照https://github.com/stanford-cs336/assignment1-basics/tree/main中README的流程配置uv。

## 下载数据：

下载 TinyStories data 和 subsample of OpenWebText：
课程原生使用wget；windows下推荐使用curl。命令：
```powershell
# 创建并进入数据目录
mkdir -p data
cd data

# 下载 TinyStories
curl.exe -L -o TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl.exe -L -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# 下载 OpenWebText (OWT)
curl.exe -L -o owt_train.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
curl.exe -L -o owt_valid.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

# 解压 .gz 文件
# Windows 10/11 自带 tar.exe，可以用来替代 gunzip
tar.exe -xvzf owt_train.txt.gz
tar.exe -xvzf owt_valid.txt.gz

cd ..
```

如果解压不成功，考虑使用python原生解压。
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
你将构建训练标准 Transformer 语言模型（LM）所需的所有组件，并训练一些模型。

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


### Problem (unicode1):
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



### Problem (unicode2):

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
数出byte pair的出现频率：
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

#### 1 