say something...
# 4 Training a Transformer

我们现在已经有了预处理数据（通过分词器）和模型（Transformer）的步骤。剩下的工作是构建支持训练的所有代码。这由以下部分组成：

• 损失函数：我们需要定义损失函数（交叉熵）。
• 优化器：我们需要定义优化器来最小化这个损失（AdamW）。
• 训练循环：我们需要所有支持性的基础设施，用于加载数据、保存检查点并管理训练。
<br> <br> <br>
## 4.1 交叉熵损失

回想一下，Transformer 语言模型为每个长度为 $m + 1$ 的序列 $x$ 以及 $i = 1, \dots, m$ 定义了一个分布 $p_\theta(x_{i+1} | x_{1:i})$。给定一个由长度为 $m$ 的序列组成的训练集 $D$，我们定义标准交叉熵（负对数似然）损失函数：

$$\ell(\theta; D) = \frac{1}{|D|m} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} | x_{1:i}). \quad (16)$$

（注意，Transformer 中的单次前向传递会产生所有 $i = 1, \dots, m$ 的 $p_\theta(x_{i+1} | x_{1:i})$。）

特别地，Transformer 为每个位置 $i$ 计算逻辑值 $o_{i} \in \mathbb{R}^{\text{vocabsize}}$，其结果为：

$$p(x_{i+1} | x_{1:i}) = \text{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocabsize}} \exp(o_i[a])}. \quad (17)$$

交叉熵损失通常是相对于逻辑值向量 $o_{i} \in \mathbb{R}^{\text{vocabsize}}$ 和目标 $x_{i+1}$ 定义的。

实现交叉熵损失需要注意数值(溢出)问题，就像 softmax 的情况一样。

### Problem (cross_entropy): 实现交叉熵

编写一个计算交叉熵损失的函数，该函数接收预测的逻辑值 ($o_{i}$) 和目标 ($x_{i+1}$)，并计算交叉熵 $\ell_i = -\log \text{softmax}(o_{i})[x_{i+1}]$。你的函数应该处理以下内容：

• 为了数值稳定性，减去最大的元素。

• 尽可能抵消 log 和 exp。

• 处理任何额外的批次维度，并返回整个批次的平均值。与 3.3 节一样，我们假设类批次维度总是出现在词表大小维度之前。

实现 [adapters.run_cross_entropy]，然后运行 uv run pytest -k test_cross_entropy 来测试。

**困惑度（Perplexity）**：交叉熵足以进行训练，但当我们评估模型时，我们也希望报告困惑度。对于长度为 $m$ 且遭受交叉熵损失 $\ell_1, \dots, \ell_m$ 的序列：

$$\text{perplexity} = \exp \left( \frac{1}{m} \sum_{i=1}^{m} \ell_i \right). \quad (18)$$
物理意义：困惑度可以被看作是模型在预测下一个词时，平均在多少个等概率的选项中做选择。$PP = 10$->模型相当于在 10 个词里犹豫。如果 $PP = |Vocab|$->模型相当于在随机选。


#### Answer ：
见transformer_training.cross_entropy和adapter.run_cross_entropy。


<br> <br> <br>
## 4.2 SGD 优化器

现在我们有了损失函数，我们将开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 $\theta_0$ 开始。然后对于每一步 $t = 0, \dots, T-1$，我们执行以下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t), \quad (19)$$

其中 $B_t$ 是从数据集 $D$ 中采样的随机数据批次，学习率 $\alpha_t$ 和批次大小 $|B_t|$ 是超参数。

### 4.2.1 在 PyTorch 中实现 SGD

为了实现我们的优化器，我们将继承 PyTorch 的 `torch.optim.Optimizer` 类。一个优化器子类必须实现两个方法：

`def __init__(self, params, ...)` 应该初始化你的优化器。这里，`params` 将是要优化的参数集合（或参数组，以防用户想要对模型的不同部分使用不同的超参数，如学习率）。确保将 `params` 传递给基类的 `__init__` 方法，该方法将存储这些参数以供在 `step` 中使用。你可以根据优化器接收额外的参数（例如，学习率是一个常见的参数），并以字典的形式将它们传递给基类构造函数，其中键是你为这些参数选择的名称（字符串）。

`def step(self)` 应该对参数进行一次更新。在训练循环期间，这将在反向传播之后被调用，因此你可以访问上一个批次的梯度。此方法应遍历每个参数张量 `p` 并就地修改它们，即根据梯度 `p.grad`（如果存在）设置 `p.data`。

PyTorch 优化器 API 有一些微妙之处，因此通过示例来解释更容易。为了使我们的示例更丰富，我们将实现 SGD 的一个略微变体，其中学习率在训练过程中衰减，从初始学习率 $\alpha$ 开始，随着时间的推移采取连续更小的步长：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t) \quad (20)$$

让我们看看这个版本的 SGD 如何实现为 PyTorch 优化器：
```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # 获取与 p 相关的状态。
                t = state.get("t", 0) # 从状态中获取迭代次数，或初始值。
                grad = p.grad.data # 获取损失相对于 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad # 就地更新权重张量。
                state["t"] = t + 1 # 增加迭代次数。
        return loss
```


在 `__init__` 中，我们将参数以及默认超参数传递给基类构造函数。在 `step` 中，我们遍历每个参数组，然后遍历该组中的每个参数，并应用公式 20。在这里，我们将迭代次数保留为与每个参数相关的状态：我们首先读取该值，在梯度更新中使用它，然后更新它。API 规定用户可以传入一个可调用的 closure 来在优化器步骤之前重新计算损失。我们不需要这个，但我们添加它以符合 API。

我们可以使用以下极简的训练循环示例来查看其运行情况：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad() # 重置所有可学习参数的梯度。
    loss = (weights**2).mean() # 计算标量损失值。
    print(loss.cpu().item())
    loss.backward() # 运行反向传播，计算梯度。
    opt.step() # 运行优化器步骤。
```


这是训练循环的典型结构：在每次迭代中，我们将计算损失并运行优化器的步骤。当训练语言模型时，我们的可学习参数将来自模型（在 PyTorch 中，`m.parameters()` 为我们提供此集合）。损失将在采样的批次数据上计算，但训练循环的基本结构将保持不变。

#### Problem (learning_rate_tuning): 调整学习率 (1 point)

正如我们将看到的，影响训练最多的超参数之一是学习率。让我们在我们的玩具示例中实践一下。使用另外三个学习率值运行上述 SGD 示例：1e1、1e2 和 1e3，仅进行 10 次训练迭代。对于这些学习率中的每一个，损失会发生什么变化？它是衰减得更快、更慢，还是发散（即在训练过程中增加）？

##### Answer：
如下，全都发散？
```powershell
PS D:\cs336_assignments_and_notes\assignment1-basics\cs336_basics> uv run python lr_tuning_test.py
===== Testing LR: 10.0 =====
Step 0: Loss = 2.50e+01
Step 1: Loss = 9.02e+03
Step 2: Loss = 1.56e+06
Step 3: Loss = 1.73e+08
Step 4: Loss = 1.40e+10
Step 5: Loss = 8.86e+11
Step 6: Loss = 4.55e+13
Step 7: Loss = 1.96e+15
Step 8: Loss = 7.22e+16
Step 9: Loss = 2.32e+18

===== Testing LR: 100.0 =====
Step 0: Loss = 2.50e+01
Step 1: Loss = 9.90e+05
Step 2: Loss = 1.95e+10
Step 3: Loss = 2.56e+14
Step 4: Loss = 2.51e+18
Step 5: Loss = 1.96e+22
Step 6: Loss = 1.28e+26
Step 7: Loss = 7.10e+29
Step 8: Loss = 3.45e+33
Step 9: Loss = 1.49e+37

===== Testing LR: 1000.0 =====
Step 0: Loss = 2.50e+01
Step 1: Loss = 9.99e+07
Step 2: Loss = 2.00e+14
Step 3: Loss = 2.66e+20
Step 4: Loss = 2.65e+26
Step 5: Loss = 2.12e+32
Step 6: Loss = 1.41e+38
Step 7: Loss = inf
Step 8: Loss = inf
Step 9: Loss = inf
```

<br> <br> <br>
## 4.3 AdamW

现代语言模型通常使用更复杂的优化器，而不是 SGD。最近使用的大多数优化器都是 Adam 优化器 [Kingma and Ba, 2015] 的衍生品。我们将使用 AdamW [Loshchilov and Hutter, 2019]，它在最近的工作中被广泛使用。AdamW 提出了一种对 Adam 的修改，通过添加权重衰减（在每次迭代中，我们将参数向 0 拉动）来改进正则化，其方式与梯度更新解耦。我们将按照 Loshchilov and Hutter [2019] 的算法 2 中描述的方式实现 AdamW。

AdamW 是有状态的：对于每个参数，它都会跟踪其一阶和二阶矩的运行估计。因此，AdamW 使用额外的内存以换取更好的稳定性和收敛性。除了学习率 $\alpha$ 外，AdamW 还有一对控制矩估计更新的超参数 ($\beta_1, \beta_2$)，以及权重衰减率 $\lambda$。算法可以写成如下形式，其中 $\epsilon$ 是用于在二阶矩 $v$ 极其小时提高数值稳定性的微小值：

### 算法  AdamW 优化器

init($\theta$)（初始化可学习参数）
$m \leftarrow 0$（一阶矩向量的初始值；形状与 $\theta$ 相同）
$v \leftarrow 0$（二阶矩向量的初始值；形状与 $\theta$ 相同）
for $t = 1, \dots, T$   do
	采样数据批次 $B_t$
	$g \leftarrow \nabla_\theta \ell(\theta; B_t)$（计算当前步的损失梯度）
	$m \leftarrow \beta_1 m + (1 - \beta_1)g$（更新一阶矩估计）
	$v \leftarrow \beta_2 v + (1 - \beta_2)g^2$（更新二阶矩估计）
	$\alpha_t \leftarrow \alpha \frac{\sqrt{1-(\beta_2)^t}}{1-(\beta_1)^t}$（计算迭代 $t$ 的调整后的 $\alpha$）
	$\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v}+\epsilon}$（更新参数）
	$\theta \leftarrow \theta - \alpha \lambda \theta$（应用权重衰减）
end for

注意 $t$ 从 1 开始。  
你现在将实现这个优化器。

### Problem (adamw): 实现 AdamW (2 points)

将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接收学习率 $\alpha$，以及 $\beta$、$\epsilon$ 和 $\lambda$ 超参数。为了帮助你保持状态，基类 `Optimizer` 为你提供了一个字典 `self.state`，它将 `nn.Parameter` 对象映射到一个字典，该字典存储该参数所需的任何信息。实现 [adapters.get_adamw_cls] 并确保它通过 `uv run pytest -k test_adamw`。

#### Answer：
见transformer_training.AdamW 。

### Problem (adamwAccounting): 使用 AdamW 训练的资源核算 (2 points)

让我们计算运行 AdamW 需要多少内存和计算量。假设我们为每个张量使用 float32。

(a) 运行 AdamW 需要多少峰值内存？根据参数、激活值、梯度和优化器状态（optimizer state）的内存使用情况分解你的答案。用 batch_size 和模型超参数 (vocab_size, context_length, num_layers, d_model, num_heads) 来表达你的答案。假设 $d_{ff} = 4 \times d_{model}$。

为了简单起见，在计算激活值的内存使用时，仅考虑以下组件：

• Transformer 块
	– RMSNorm(s)
	– 多头自注意力子层：QKV 投影、$Q^\top K$ 矩阵乘法、softmax、值的加权和、输出投影。
	– 逐位置前馈：$W_1$ 矩阵乘法、SiLU、$W_2$ 矩阵乘法

• 最终 RMSNorm
• 输出嵌入
• 逻辑值上的交叉熵

回答：参数、激活值、梯度和优化器状态以及总和的代数表达式。

(b) 为 GPT-2 XL 形状的模型实例化你的答案，得到一个仅依赖于 batch_size 的表达式。在 80GB 内存范围内，你可以使用的最大批次大小是多少？

回答：一个看起来像 $a \cdot \text{batchsize} + b$ 的数值表达式，以及代表最大批次大小的数字。

(c) 运行 AdamW 的一个步骤需要多少个 FLOPs？


(d) 模型 FLOPs 利用率 (MFU) 定义为观察到的吞吐量（每秒 token 数）与硬件理论峰值 FLOP 吞吐量的比值 [Chowdhery et al., 2022]。NVIDIA A100 GPU 对于 float32 操作的理论峰值为 19.5 teraFLOP/s。假设你能够获得 50% 的 MFU，在单台 A100 上训练 GPT-2 XL 400K 步且批次大小为 1024 需要多长时间？遵循 Kaplan et al. [2020] 和 Hoffmann et al. [2022]，假设反向传播的 FLOPs 是前向传播的两倍。

回答：训练所需的天数及简要说明。

#### Answer：

(a)计算峰值内存
A. 静态：
设参数个数N。
内存里：参数N个，梯度N个，optimizer state2N个(m和v)   
共4N。
又，由chapter3最后一问的结尾，有$N \approx vocab\_size \cdot d_{model}+ num\_layers \cdot (12d_{model}^2) + vocab\_size \cdot d_{model}$
B. 动态：
1 每层2个ln：$2 \times (B \cdot T \cdot d_{model})$
2 QKV投影：需要存输入X，即$B \cdot T \cdot d_{model}$。
3 $Q^\top K$ : 存 Q和K，$2 \times (B \cdot T \cdot d_{model})$。
4 softmax：梯度(p-y)，只用存p，$B \cdot num\_heads \cdot T \cdot T$。
5 value的加权和（P* V)：存p（已存）和V：$B \cdot T \cdot d_{model}$。
6 输出投影：存加权和的结果，$B \cdot T \cdot d_{model}$。
7 FFN：i 乘W1（W3）用到X，大小$B \cdot T \cdot d_{model}$。ii $Y_{inter} = \text{SiLU}(Y_1) \odot Y_3$ 用到Y1，Y3， $\text{SiLU}(Y_1)$， 大小$3 \times (B \cdot T \cdot d_{ff})$。   iii W2 * $Y_{inter}$ 用到$Y_{inter}$， 大小$B \cdot T \cdot d_{ff}$。
8 最终ln + 输出嵌入：$2 \cdot (BTd_{model}) = 2BTd_{model}$
9 交叉熵：存logits，大小$BT \cdot vocab\_size$

综上，相加，乘4bytes。得到：$Memory_{peak} = 16N + L \cdot (96BTd_{model} + 4BhT^2) + 4BT(2d_{model} + vocab\_size)$ 

(b) 对于 GPT-2 XL ($d_{model}=1600, L=48, vocab=50257$): $N \approx 2 \cdot (1600 \cdot 50257) + 48 \cdot 16 \cdot 1600^2 \approx 2.12 \times 10^9$。
静态内存16N = 33.92GB。最终层0.22GB。
transformer共12.8GB。因此80GB内存下，所求计算式为33.92 + batchsize * 12.8 。 batchsize ≤ [3.6] = 3

(c)  用ch3最后一问的结论，$F_{total} = 4.5 \text{ TFLOPs}$， 总 FLOPs 为 **$13.5 \text{ TFLOPs}$

(d) $400,000 \text{step} \times 1024 \text{ Batch} \times 13.5 \text{ TFLOPs/step} \approx 5.53 \times 10^{21} \text{ FLOPs}$。
$5.53 \times 10^{21} \div (9.75 \times 10^{12}) \div 86400 \approx$ 6564 天。
<br> <br> <br>

## 4.4 学习率调度(scheduling)

导致损失下降最快的学习率值在训练过程中通常会发生变化。在训练 Transformer 时，典型做法是使用学习率调度，我们在开始时使用较大的学习率，进行较快的更新，并随着模型训练缓慢衰减到一个较小的值。在本作业中，我们将实现用于训练 LLaMA 的余弦退火。

一个调度器仅仅是一个函数，它接收当前步长 $t$ 和其他相关参数（如初始和最终学习率），并返回梯度更新在步骤 $t$ 使用的学习率。最简单的调度是常数函数。

余弦退火学习率调度接收: (i) 当前迭代 $t$，(ii) 最大学习率 $\alpha_{\max}$，(iii) 最小（最终）学习率 $\alpha_{\min}$，(iv) 预热迭代次数 $T_w$，以及 (v) 余弦退火迭代次数 $T_c$。迭代 $t$ 的学习率定义为：

(预热) 如果 $t < T_w$，则 $\alpha_t = \frac{t}{T_w} \alpha_{\max}$。

(余弦退火) 如果 $T_w \le t \le T_c$，则 $\alpha_t = \alpha_{\min} + \frac{1}{2} \left( 1 + \cos \left( \frac{t-T_w}{T_c-T_w} \pi \right) \right) (\alpha_{\max} - \alpha_{\min})$。

(后退火) 如果 $t > T_c$，则 $\alpha_t = \alpha_{\min}$。

### Problem (learning_rate_schedule): 实现带预热的余弦学习率调度

编写一个函数，接收 $t, \alpha_{\max}, \alpha_{\min}, T_w$ 和 $T_c$，并根据上面定义的调度器返回学习率 $\alpha_t$。然后实现 [adapters.get_lr_cosine_schedule] 并确保它通过 uv run pytest -k test_get_lr_cosine_schedule。

#### Answer：
见transformer_training.lr_cosine_schedule和adapter.run_get_lr_cosine_schedule。
<br> <br> <br>
## 4.5 梯度裁剪

在训练期间，我们有时会遇到产生巨大梯度的训练样本，这可能会使训练不稳定。为了缓解这种情况，实践中经常采用的一种技术是梯度裁剪。这个想法是在每轮反向传播之后、采取优化器步骤之前，对梯度的范数实施限制。

给定梯度（针对所有参数） $g$，我们计算其 $\ell_2$-范数 $\|g\|_2$。如果该范数小于最大值 $M$，则保持 $g$ 不变；否则，我们将 $g$ 按比例 $\frac{M}{\|g\|_2+\epsilon}$ 缩小（其中添加了一个微小的 $\epsilon$，如 $10^{-6}$，以保证数值稳定性）。注意，结果范数将略低于 $M$。

### Problem (gradient_clipping): 实现梯度裁剪 (1 point)

编写一个实现梯度裁剪的函数。你的函数应接收参数列表和最大 $\ell_2$-范数。它应就地修改每个参数的梯度。使用 $\epsilon = 10^{-6}$（PyTorch 默认值）。然后，实现适配器 [adapters.run_gradient_clipping] 并确保它通过 uv run pytest -k test_gradient_clipping。

#### Answer：
见transformer_training.gradient_clipping和adapter.run_gradient_clipping。