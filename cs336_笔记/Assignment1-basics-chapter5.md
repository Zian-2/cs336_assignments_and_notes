## 5 Training Loop

现在将目前构建的主要组件组合在一起：tokenized data，模型和优化器。

### 5.1 数据加载器

tokneized data（例如，你在 `tokenizer_experiments` 中准备的数据）是一个单一的标记序列 $x = (x_1, . . . , x_n)$。尽管源数据可能由单独的文档组成（例如，不同的网页或源代码文件），通用的做法是将所有这些文档连接成一个单一的标记序列，并在它们之间添加分隔符（如 `<|endoftext|>` 标记）。

数据加载器将此序列转换为批处理流，其中每个批次由 `B` 个长度为 `m` 的序列组成，并配对相应的长度也为 `m` 的下一个标记。例如，对于 `B = 1`, `m = 3`，`([x2, x3, x4], [x3, x4, x5])` 就是一个潜在的批次。

以这种方式加载数据可以从多个方面简化训练。首先，任何 $1 \le i < n - m$ 都能提供一个有效的训练序列，因此采样序列变得非常简单。由于所有训练序列的长度相同，不需要对输入序列进行填充（padding），这提高了硬件利用率（也通过增加批次大小 `B` 实现）。最后，我们也不需要完全加载整个数据集来采样训练数据，这使得处理那些无法完全放入内存的大型数据集变得容易。

### Problem (data_loading)：实现数据加载 (2 分)

编写一个函数，接收一个 `numpy` 数组 `x`（带有标记 ID 的整数数组）、`batch_size`、`context_length` 和一个 `PyTorch` 设备字符串（例如 `'cpu'` 或 `'cuda:0'`），并返回一对张量：采样的输入序列和相应的下一标记目标。两个张量的形状都应为 `(batch_size, context_length)`，包含标记 ID，并且都应放置在请求的设备上。为了针对我们提供的测试来测试你的实现，你首先需要实现位于 `[adapters.run_get_batch]` 的测试适配器。然后，运行 `uv run pytest -k test_get_batch` 来测试你的实现。

低资源/缩减规模提示：在 `CPU` 或 `Apple Silicon` 上进行数据加载。

如果你计划在 `CPU` 或 `Apple Silicon` 上训练你的语言模型，你需要将数据移动到正确的设备上（同样地，你稍后也应该为你的模型使用相同的设备）。如果你在 `CPU` 上，可以使用 `'cpu'` 设备字符串；如果你在 `Apple Silicon`（M* 芯片）上，可以使用 `'mps'` 设备字符串。

有关 `MPS` 的更多信息，请查看以下资源：

[https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)  

[https://pytorch.org/docs/main/notes/mps.html](https://pytorch.org/docs/main/notes/mps.html)

#### Answer ：
见transformer_training.get_batch和adapter.run_get_batch。  


如果数据集太大无法加载到内存中怎么办？我们可以使用一个名为 `mmap` 的 Unix 系统调用，它将磁盘上的文件映射到虚拟内存，并在访问该内存位置时延迟加载文件内容。因此，你可以“假装”整个数据集都在内存中。`Numpy` 通过 `np.memmap` 实现这一点（或者如果在最初保存数组时使用了 `np.save`，则在 `np.load` 中使用 flag `mmap_mode='r'`），这将返回一个类似于 `numpy` 数组的对象，在你访问条目时按需加载。在训练期间从数据集（即 `numpy` 数组）采样时，请确保以内存映射模式加载数据集（通过 `np.memmap` 或 `np.load` 的 `mmap_mode='r'` 标志，取决于你保存数组的方式）。确保你还指定了与正在加载的数组匹配的 `dtype`。显式验证内存映射数据看起来是否正确（例如，不包含超出预期词表大小的值）可能会很有帮助。

### 5.2 检查点 (Checkpointing)

除了加载数据，我们在训练时还需要保存模型。运行作业时，我们通常希望能够恢复因某种原因中途停止的训练运行（例如，由于作业超时、机器故障等）。即使一切顺利，我们也可能希望稍后能够访问中间模型（例如，为了事后研究训练动态、从不同训练阶段的模型中获取样本等）。

一个检查点应该包含我们恢复训练所需的所有状态。我们当然希望至少能够恢复模型权重。如果使用有状态的优化器（如 `AdamW`），我们还需要保存优化器的状态（例如，在 `AdamW` 的情况下，即动量估计）。最后，为了恢复学习率调度，我们需要知道停止时的迭代次数。`PyTorch` 使得保存所有这些变得很容易：每个 `nn.Module` 都有一个 `state_dict()` 方法，返回一个包含所有可训练权重的字典；我们可以稍后使用姊妹方法 `load_state_dict()` 来恢复这些权重。对于任何 `nn.optim.Optimizer` 也是如此。最后，`torch.save(obj, dest)` 可以将一个对象（例如，在某些值中包含张量的字典，也可以是像整数这样的常规 `Python` 对象）转储到文件（路径）或类文件对象中，然后可以使用 `torch.load(src)` 将其加载回内存。

### Problem (checkpointing)：实现模型检查点 (1 分)

实现以下两个函数来加载和保存检查点：  

`def save_checkpoint(model, optimizer, iteration, out):`
应该将前三个参数的所有状态转储到类文件对象 out 中。你可以使用模型和优化器的 state_dict 方法来获取它们的相关状态，并使用 torch.save(obj, out) 转储到 out 中。

该函数预期以下参数：  

`model`: `torch.nn.Module`  

`optimizer`: `torch.optim.Optimizer`  

`iteration`: `int`    

`out`: `str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`  

`def load_checkpoint(src, model, optimizer):`应该从 src（路径或类文件对象）加载检查点，然后从该检查点恢复模型和优化器状态。你的函数应该返回保存在检查点中的迭代次数。

该函数预期以下参数：   

`src`: `str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`  

`model`: `torch.nn.Module`  

`optimizer`: `torch.optim.Optimizer`  

实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` 适配器，并确保它们通过 `uv run pytest -k test_checkpointing`。

#### Answer ：
见transformer_training.save(load)_ checkpoint和adapter.run_save(load)_ checkpoint。  

### 5.3 训练循环

现在，终于可以将你实现的所有组件组合到你的主训练脚本中了。使训练运行易于启动并配置不同的超参数（例如，通过将它们作为命令行参数）是值得的，因为你稍后将多次执行这些操作，以研究不同选择如何影响训练。

### Problem (training_together)：(4 分)

编写一个脚本，运行训练循环以在用户提供的输入上训练你的模型。特别是，我们建议你的训练脚本至少允许以下操作：

能够配置和控制各种模型和优化器超参数。  
使用 `np.memmap` 高效内存加载训练和验证大型数据集。  
将检查点序列化到用户提供的路径。  
定期记录训练和验证性能（例如，记录到控制台和/或外部服务，如 `Weights and Biases`）。  
#### Answer ：
见transformer_training_loop.py