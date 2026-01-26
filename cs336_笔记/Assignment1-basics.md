# 1 作业总览
你将构建训练标准 Transformer 语言模型（LM）所需的所有组件，并训练一些模型。

### 你将实现：

1 BPE 分词器 (Byte-pair encoding tokenizer)：（第 2 节） 
2 Transformer 语言模型 (LM)：（第 3 节） 
3 交叉熵损失函数与 AdamW 优化器：(第 4 节) 
4 训练循环：支持模型和优化器状态的序列化与加载（保存与读取）:（第 5 节） 


### 你将完成如下任务：

1 在 TinyStories 数据集上训练一个 BPE 分词器。 
2 对数据集运行训练好的分词器，将其转换为整数 ID 序列。 
3 在 TinyStories 数据集上训练 Transformer 语言模型。 
4 使用训练好的模型生成样本并评估困惑度 (Perplexity)。 
5 在 OpenWebText 数据集上训练模型，并将达到的困惑度结果提交到排行榜。 
 

### 允许使用的工具：

课程希望你从0开始搭组件，所以你不得使用`torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义，除了：

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




# 2 BPE(Byte-Pair Encoding，字节对编码)分词器

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


### 课程文档外的补充：UTF-8编码规则

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


## 2.3 子词分词方案（subword tokenization）

将词语拆成byte序列之后，仍然不能逐个byte拆分来看作为token：这会导致序列变得极长，训练的step变长，计算量也增大（如transformer计算复杂度与序列长度成正比）；同时更长的序列也导致信息密度被稀释，网络寻找token间的关系变得更困难。

而subword tokenization就是指代一个折中的方案。它选择用一个更大的词汇表(vocabulary)来trade-off更短的序列。比如说，如果‘the’经常出现，我们就可以给它单独分配一个条目(entry)，词汇表维度+1，但把3个token缩短成了1个。

为了做这样的工作，Sennrich et al. (2016) 提出使用byte-pair encoding (BPE; Gage, 1994)。作为一种subword tokenization，它简单地基于出现频率，将经常出现的byte pair合并(merge)成一个未被使用的索引。基于BPE的subword tokenizer被称为BPE tokenizer。


## 2.4 训练BPE分词器




