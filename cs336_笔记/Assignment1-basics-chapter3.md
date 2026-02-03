say something
# 3  Transformer 架构

一个语言模型接受一批整数token ID序列作为输入（即，形状为`(batch_size, sequence_length)` 的 `torch.Tensor`），并返回词表上的（批量）归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch 张量），其中预测的分布是针对每个输入 token 的下一个词的。在训练语言模型时，我们使用这些下文预测来计算实际下一个词与预测下一个词之间的交叉熵损失。在推理过程中利用语言模型生成文本时，我们取最后一个时间步（即序列中的最后一项）预测的下文概率分布来生成序列中的下一个 token（例如，通过选取概率最高的 token、从分布中采样等），将生成的 token 添加到输入序列中，并重复此过程。

在作业的这一部分中，你将从零开始构建这个 Transformer 语言模型。我们将先从模型的高层描述开始，然后逐步详细介绍各个组件。

### 3.1 Transformer 语言模型（LM）

给定一个 token ID 序列，Transformer 语言模型使用输入嵌入（input embedding）将 token ID 转换为稠密向量，将嵌入后的 token 通过 `num_layers` 个 Transformer 块，然后应用一个学习到的线性投影（“输出嵌入”或“LM head”）来产生预测的下一 token 的 logits。示意图请参见图 1。

#### 3.1.1 Token 嵌入

在第一步中，Transformer 将（批量的）token ID 序列嵌入为包含 token 身份信息的向量序列（图 1 中的红色块）。

具体来说，给定一个 token ID 序列，Transformer 语言模型使用一个 token 嵌入层来生成一个向量序列。每个嵌入层接收一个形状为 `(batch_size, sequence_length)` 的整数张量，并产生一个形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

#### 3.1.2 Pre-norm Transformer 块

嵌入之后，激活值由几个结构相同的神经网络层处理。一个标准的纯解码器（decoder-only）Transformer 语言模型由 `num_layers` 个相同的层（通常称为 Transformer “块”）组成。每个 Transformer 块接收形状为 `(batch_size, sequence_length, d_model)` 的输入，并返回形状为 `(batch_size, sequence_length, d_model)` 的输出。每个块通过自注意力机制（self-attention）聚合整个序列的信息，并通过前馈层（feed-forward layers）对其进行非线性转换。

### 3.2 输出归一化与嵌入

在经过 `num_layers` 个 Transformer 块之后，我们将取最终的激活值并将其转化为词表上的分布。

我们将实现“Pre-norm” Transformer 块（详见 §3.5），这额外要求在最后一个 Transformer 块之后使用层归一化（Layer Normalization，详见下文），以确保其输出被正确缩放。

在此归一化之后，我们将使用一个标准的学习线性变换，将 Transformer 块的输出转换为预测的下一 token 的 logits（例如，参见 Radford 等人 [2018] 的公式 2）。

### 3.3 备注：批处理、Einsum 与高效计算

在整个 Transformer 中，我们将对许多类似批次的输入执行相同的计算。以下是一些示例：

- **批次元素**：我们对每个批次元素应用相同的 Transformer 前向操作。
    
- **序列长度**：像 RMSNorm 和前馈层这样的“位置级（position-wise）”操作对序列的每个位置执行完全相同的操作。
    
- **注意力头**：在“多头”注意力操作中，注意力操作跨注意力头进行批处理。
    

拥有一个符合人体工程学的方式来执行这些操作是很有用的，这样既能充分利用 GPU，又易于阅读和理解。许多 PyTorch 操作可以在张量开头接收多余的“类批次（batch-like）”维度，并高效地跨这些维度重复/广播操作。

例如，假设我们正在做一个位置级的批量操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的“数据张量” $D$，我们想对一个形状为 `(d_model, d_model)` 的矩阵 $A$ 进行批量向量-矩阵乘法。在这种情况下，`D @ A` 将执行批量矩阵乘法，这是 PyTorch 中一个高效的原语，其中 `(batch_size, sequence_length)` 维度被批量处理。

因此，假设你的函数可能会被赋予额外的类批次维度，并将这些维度保持在 PyTorch 形状的开头是有帮助的。为了以这种方式组织张量以便进行批处理，可能需要使用 `view`、`reshape` 和 `transpose` 的多个步骤来调整它们。这可能有点痛苦，而且通常很难读懂代码在做什么以及张量的形状是什么。

一个更符合人体工程学的选择是在 `torch.einsum` 中使用 einsum 符号，或者使用像 `einops` 或 `einx` 这样与框架无关的库。两个核心操作是 `einsum`（可以对输入张量的任意维度进行张量收缩）和 `rearrange`（可以对任意维度进行重排、连接和拆分）。事实证明，机器学习中几乎所有的操作都是维度操作和张量收缩的某种组合，偶尔带有（通常是逐元素的）非线性函数。这意味着使用 einsum 符号可以使你的很多代码更具可读性和灵活性。

我们强烈建议在课程中学习并使用 einsum 符号。以前没有接触过 einsum 符号的学生应该使用 `einops`（文档见此处），而已经熟悉 `einops` 的学生应该学习更通用的 `einx`（见此处）。这两个包都已经安装在我们提供的环境中。

这里我们给出一些如何使用 einsum 符号的例子。这些是对 `einops` 文档的补充，你应该先阅读文档。