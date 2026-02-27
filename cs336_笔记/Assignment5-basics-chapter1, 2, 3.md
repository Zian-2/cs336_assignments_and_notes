# 1 Overview
作业5中你将动手训练推理模型来推理数学问题。

**你将实现：**
1. 基于MATH数据集（Hendrycks et al.）的零样本提示词基准（zero-shot prompting baseline）。
2. 使用更强的推理模型（DeepSeek R1, DeepSeekAI et al. 2025）的推理轨迹，进行监督微调（SFT）。
3. 专家迭代（Expert Iteration）来通过经验证的奖励来提升推理性能。
4.  组相对策略优化（Group-Relative Policy Optimization , GRPO）来通过经验证的奖励来提升推理性能。

**你将运行：**
1. 测量 Qwen 2.5 Math 1.5B 的零样本提示词性能（作为基准）。
2. 使用来自 R1 的推理迹线对 Qwen 2.5 Math 1.5B 进行监督微调。
3. 使用经过验证的奖励在 Qwen 2.5 Math 1.5B 上运行专家迭代。
4.  使用经过验证的奖励在 Qwen 2.5 Math 1.5B 上运行 GRPO。

**代码结构：**
所有作业代码以及本说明文档均可在 GitHub 上获取： [github.com/stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment)
1. cs336_alignment/* ：这是你编写assignment 5 代码的地方。请注意，这里除了少许初始代码外没有其他代码，因此你可以从零开始进行任何操作。  
2. cs336_alignment/prompts/* ：为了方便起见，我们提供了包含提示词的文本文件，以尽量减少从 PDF 复制粘贴提示词到代码中可能产生的错误。
3. tests/* .py：这包含了你必须通过的所有测试。你只需要通过 tests/test_sft.py 和 tests/test_grpo.py 中的测试——其余测试针对作业的非强制部分。这些测试调用了 tests/adapters.py 中定义的钩子。你将实现adapters以将你的代码连接到测试。编写更多测试和/或修改测试代码可能对调试有所帮助，但你的实现应当通过原始提供的测试套件。
4. README.md：该文件包含有关设置环境的一些基本说明。

**可以使用的工具：**
我们希望你从头开始构建大部分与强化学习（RL）相关的组件。你可以使用 vLLM 等工具从语言模型生成文本（§3.1）。此外，你可以使用 HuggingFace Transformers 来加载 Qwen 2.5 Math 1.5B 模型及分词器并运行前向传播（§4.1），但你不得使用任何训练工具类（例如 Trainer 类）。

**如何提交：**  
你将向 Gradescope 提交以下文件：
writeup.pdf：回答所有书面问题。请对你的回答进行排版。
code.zip：包含你编写的所有代码。


# 2 Reasoning with Language Models

## 2.1 Motivation

语言模型最卓越的使用案例构建能够处理广泛自然语言处理任务的通用系统。本次作业中，我们将专注于一个新兴用例：数学推理。它将作为我们设置评估、执行SFT以及实验利用强化学习教导语言模型推理的试验场。

这与我们之前的作业有两个不同之处。

第一，我们不会使用之前作业中的语言模型代码库和模型。理想地，我们希望使用之前作业训练的基础语言模型，但微调这些模型不会得到令人满意的结果——这些模型过于薄弱，无法展现出非平凡的数学推理能力。因此，我们将切换到一个我们可以访问的现代高性能语言模型（Qwen 2.5 Math 1.5B Base），并在该模型基础上完成大部分工作。

第二，我们将引入一个新的基准来评估我们的语言模型。直到现在，我们一直接受交叉熵作为下游任务良好的指标。然而，本次作业的重点是弥合基础模型与下游任务之间的差距，因此我们必须使用独立于交叉熵的评估方法。我们将使用来自 Hendrycks 等人（2021）的 MATH 12K 数据集，该数据集由具有挑战性的高中数学竞赛题目组成。我们将通过把语言模型的输出与参考答案进行比较来评估其表现。

## 2.2 思维链（Chain-of-Thoughts, CoTs）推理与推理强化学习

语言模型领域最近一个令人兴奋的趋势是使用思维链推理来提高在各种任务上的表现。思维链是指通过逐步思考问题的过程，在得出最终答案之前生成中间推理步骤。

**使用大语言模型进行思维链推理：**  
早期的思维链方法通过使用“草稿本（scratchpad）"将问题分解为中间步骤，从而微调语言模型以解决算数等简单的数学任务（Nye et al.，2021）。其他研究提示强大模型在回答之前“一步步思考”，发现这能显著提高在小学数学题等数学推理任务上的表现（Wei et al.，2023）。

**通过专家迭代学习推理：**  
自学推理者（STaR）（Zelikman et al.，2022）将推理构建为一个自引导的循环(bootstrapping loop)：预训练模型首先采样多样化的思维链（CoTs），仅保留那些引导至正确答案的思维链，然后在这些“专家”迹线上进行微调。迭代这一循环可以提高语言模型的推理能力和解题率。STaR 证明了这种形式的专家迭代（Anthony et al.，2017），利用基于字符串匹配的生成答案自动验证，可以在没有人工编写推理迹线的情况下引导推理技能。

**带有验证奖励的推理强化学习、o1 和 R1：**  
最近的研究探索了使用更强大的强化学习算法和验证奖励来提高推理性能。OpenAI 的 o1（以及随后的 o3/o4）（OpenAI et al.，2024）、DeepSeek 的 R1（DeepSeek-AI et al.，2025）以及 Moonshot 的 kimi k1.5（Team et al.，2025）使用策略梯度方法（Sutton et al.，1999），在字符串匹配或单元测试可验证正确性的数学和代码任务上进行训练，在竞赛数学和编程表现上展示了显著的进步。随后的研究如 Open-R1（Face, 2025）、SimpleRL-Zoo（Zeng et al.，2025）和 TinyZero（Pan et al.，2025）确认，纯粹的带有验证奖励的强化学习——即使是在 1.5B 参数规模的小模型上——也能提高推理性能。

**Our Setup：模型与数据集：**  
在接下来的章节中，我们将考虑复杂性由浅入深的方法来训练基础语言模型逐步推理以解决数学问题。在本次作业中，我们将使用 Qwen 2.5 Math 1.5B Base 模型，该模型是在高质量合成数学预训练数据上从 Qwen 2.5 1.5B 模型持续预训练而来的（Yang et al.，2024）。MATH 数据集可在 Together 集群的 /data/a5-alignment/MATH 路径获取。

**给旁听生的提示：替代数据集：**  
不幸的是，由于版权声明，MATH 数据集无法公开获取。如果你是在校外跟随练习，可以使用以下开源数学推理数据集之一：

- Countdown（Pan et al.，2025）：一个基于英国电视节目 Countdown 的简单合成任务，已成为小规模推理强化学习的流行试验场。
- GSM8K（Cobbe et al.，2021a）：小学数学题，比 MATH 简单，但可以让你调试正确性并熟悉推理强化学习流水线。
- Tulu 3 SFT Math（Lambert et al.，2025）：使用 GPT-4o 和 Claude 3.5 Sonnet 生成的合成数学题。由于这些是合成的，某些答案（甚至问题本身）可能不完全正确。
- 此处链接的其他数学 SFT 数据集。

如果未直接提供简短的ground-truth（例如 1/2），你可以使用 Math-Verify 等数学答案解析器处理 ground-truth 列。


# 3 测量Zero-shot的MATH 性能

我们将从测量基础语言模型在 MATH 数据集 5K 条示例测试集上的性能开始。建立此基准对于理解后续每种方法如何影响模型行为非常有用。

除非另有说明，在 MATH 实验中，我们将使用来自 DeepSeek R1-Zero 模型（DeepSeek-AI et al., 2025）的以下提示词。我们将其称为 r1_zero 提示词：
```
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within  <answer> </answer> tags, respectively, i.e., <think> reasoning process here  </think> <answer> answer here </answer>.
User: {question} 
Assistant: <think>
```

r1_zero 提示词位于文本文件 cs336_alignment/prompts/r1_zero.prompt 中。

在提示词中，question 指的是我们插入的某个问题（例如：Natalia 在四月向她的 48 位朋友出售了夹子，五月售出的数量是四月的一半。Natalia 在四月和五月总共卖出了多少个夹子？）。期望模型扮演助手的角色，开始生成思考过程（因为我们已经包含了一个左思考标签 `<think>`），使用 `</think>` 结束思考过程，然后在答案标签内生成最终的符号答案，如 `<answer> 4x + 10 </answer>`。让模型生成此类标签的目的是为了方便我们解析模型输出并将其与标准答案进行比较，同时在看到右答案标签 `</answer>` 时停止响应生成。

**关于提示词选择的注意：** 事实证明，由于提示词与 Qwen 2.5 Math 1.5B 模型的预训练方式不匹配，r1_zero 提示词并不是最大化强化学习（RL）后下游性能的最佳选择。Liu et al. [2025] 发现，仅用问题提示模型（不添加其他内容）初始准确率就非常高，例如，其初始表现就能匹配 r1_zero 提示词在强化学习 100 多步后的表现。他们的发现表明 Qwen 2.5 Math 1.5B 已经在类似的问答对上进行了预训练。  
尽管如此，我们本次作业仍选择 r1_zero 提示词，因为配合该提示词的强化学习在较短步数内就能显示出明显的准确率提升，这让我们能够快速走通强化学习的机制并检查正确性，即使我们最终没有达到最佳性能。作为现实对照，你将在作业后续部分直接与 question_only 提示词进行对比。

## 3.1 使用 vLLM 进行离线语言模型推理

为了评估我们的语言模型，我们需要为各种提示词生成续写（响应）。虽然完全可以自己实现生成函数（例如你在作业 1 中所做的那样），但强化学习的高效实现需要高性能的推理技术，而实现这些推理技术超出了本次作业的范围。因此，在本次作业中，我们建议使用 vLLM 进行离线批处理推理。vLLM 是一个用于语言模型的高吞吐、内存高效的推理引擎，它整合了多种有用的效率技术（如优化的 CUDA 内核、用于高效注意力 KV 缓存的 PagedAttention 等）。使用 vLLM 为一系列提示词生成续写的示例如下：

```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
	"Hello, my name is",
	"The president of the United States is",
	"The capital of France is",
	"The future of AI is",
]

# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
	temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

# Create an LLM.
llm = LLM(model=<path_to_model>)

# Generate texts from the prompts. The output is a list of RequestOutput objects 
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
	prompt = output.prompt
	generated_text = output.outputs[0].text
	print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

在上述示例中，LLM 可以使用 HuggingFace 模型的名称（如果本地未找到，将自动下载并缓存）或 HuggingFace 模型的路径进行初始化。由于下载可能耗时较长（尤其是大型模型，如 70B 参数模型），且为了节省集群磁盘空间（以免每个人都有自己的预训练模型副本），我们已在 Together 集群的以下路径下载了预训练模型。请不要在 Together 集群上重复下载这些模型：

- Qwen 2.5 Math 1.5B Base（用于推理实验）： /data/a5-alignment/models/Qwen2.5-Math-1.5B

- Llama 3.1 8B Base（用于可选的指令微调实验）： /data/a5-alignment/models/Llama-3.1-8B

- Llama 3.3 70B Instruct（用于可选的指令微调实验）： /data/a5-alignment/models/Llama-3.3-70B-Instruct


## 3.2 零样本 MATH 基准

**提示词设置。** 为了评估 MATH 测试集上的零样本性能，我们将简单地加载示例，并使用上述 r1_zero 提示词提示语言模型回答问题。

**评估指标。** 当我们评估多项选择或二元响应任务时，评估指标很明确——我们测试模型输出是否与正确答案完全一致。

在数学问题中，我们假设存在已知的标准答案（例如 0.5），但我们不能简单地测试模型是否恰好输出 0.5——它也可能回答 1/2。因此，在评估 MATH 时，我们必须解决匹配语言模型语义等价响应的棘手问题。

为此，我们希望设计某种答案解析函数，该函数将模型输出和已知标准答案作为输入，并返回一个布尔值，指示模型输出是否正确。例如，奖励函数可能会接收以 `She sold 15 clips. </answer> ` 结尾的模型字符串输出和标准答案 72，如果模型输出正确则返回 True，否则返回 False（在这种情况下，它应返回 False）。

对于我们的 MATH 实验，我们将使用近期推理强化学习研究中使用的快速且相当准确的答案解析器。该奖励函数在 `cs336_alignment.drgrpo.grader.r1_zero_reward_fn` 中实现，除非另有说明，否则你应该使用它来评估 MATH 上的性能。

**生成超参数。** 生成响应时，我们将采样温度设为 1.0，top-p 设为 1.0，最大生成长度为 1024。提示词要求模型以字符串 `</answer>` 结束其答案，因此我们可以指示 vLLM 在模型输出此字符串时停止：

```
# Based on Dr. GRPO: stop when the model completes its answer 
# https://github.com/sail-sg/understand-r1-zero/blob/ 
# c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
sampling_params.stop = ["</answer>"] sampling_params.include_stop_str_in_output = True
```

### Problem (math_baseline)：4 points

(a) 编写一个脚本来评估 Qwen 2.5 Math 1.5B 在 MATH 上的零样本性能。该脚本应 (1) 从 /data/a5-alignment/MATH/validation.jsonl 加载 MATH 验证示例，(2) 使用 r1_zero 提示词将其格式化为发送给语言模型的字符串提示词，并且 (3) 为每个示例生成输出。该脚本还应 (4) 计算评估指标，并 (5) 将示例、模型生成内容以及对应的评估分数序列化到磁盘，以便在后续问题中进行分析。

在你的实现中，包含一个类似以下参数的 evaluate_vllm 方法可能会有帮助，因为你稍后可以重复使用它：
```python
def evaluate_vllm( 
	vllm_model: LLM, 
	reward_fn: Callable[[str, str], dict[str, float]], 
	prompts: List[str], 
	eval_sampling_params: SamplingParams 
) -> None: 
	""" 
	Evaluate a language model on a list of prompts, 
	compute evaluation metrics, and serialize results to disk. 
	"""
```

提交：一个用于评估基准零样本 MATH 性能的脚本。

(b) 在 Qwen 2.5 Math 1.5B 上运行你的评估脚本。有多少个模型生成结果属于以下各类别：(1) 正确，且格式奖励和答案奖励均为 1；(2) 格式奖励为 1 且答案奖励为 0；(3) 格式奖励为 0 且答案奖励为 0？观察至少 10 个格式奖励为 0 的案例，你认为问题出在基础模型的输出还是解析器？为什么？对于（至少 10 个）格式奖励为 1 但答案奖励为 0 的案例，情况又是如何？

提交：关于模型和奖励函数表现的评论，包括各分类的示例。

(c) Qwen 2.5 Math 1.5B 零样本基准在 MATH 上的表现如何？

提交：包含评估指标的 1-2 句话。

#### Answer：
(a) 见baseline.py  
(b) 取了200个样本：
```powershell
fully_correct: 49; 
format_correct_only: 56; 
format_wrong: 95
```
示例：
```json
{

        "question": "What is $10.0000198\\cdot 5.9999985401\\cdot 6.9999852$ to the nearest whole number?",

        "ground_truth": "420",

        "model_response": "<think> To solve the problem, we need to calculate the product of the three numbers and then round the result to the nearest whole number. Let's break down the steps:\n\n1. Calculate the product of the three numbers.\n2. Round the result to the nearest whole number.\n\nWe can use Python to perform these calculations accurately. Let's write the code to do this.\n```python\n# Define the three numbers\nnum1 = 10.0000198\nnum2 = 5.9999985401\nnum3 = 6.9999852\n\n# Calculate the product\nproduct = num1 * num2 * num3\n\n# Round the product to the nearest whole number\nrounded_product = round(product)\n\nprint(rounded_product)\n```\n```output\n420\n```\nThe product of \\(10.0000198 \\cdot 5.9999985401 \\cdot 6.9999852\\) rounded to the nearest whole number is \\(\\boxed{420}\\).</answer>",

        "scores": {

            "format_reward": 0.0,

            "answer_reward": 0.0,

            "reward": 0.0

        }

    },

    {

        "question": "Two fair 6-sided dice are rolled.  What is the probability the sum rolled is 9?",

        "ground_truth": "\\dfrac{1}{9}",

        "model_response": "<think> The sum of two dice is 9 if the first die is 3 and the second die is 6, or the first die is 4 and the second die is 5, or the first die is 5 and the second die is 4, or the first die is 6 and the second die is 3.  There are 4 ways to get a sum of 9.  There are 6 possible outcomes for the first die and 6 possible outcomes for the second die, so there are 6 * 6 = 36 possible outcomes in total.  Therefore, the probability of rolling a sum of 9 is 4/36 = 1/9.  </think> <answer> 1/9  </answer>",

        "scores": {

            "format_reward": 1.0,

            "answer_reward": 1.0,

            "reward": 1.0

        }

    },

    {

        "question": "For what value of $x$ is the following equation true: $6500 + x - 4500 = 3400 + 2000$?",

        "ground_truth": "3400",

        "model_response": "<think> To find the value of $x$, we need to isolate $x$ on one side of the equation. We can do this by first subtracting 6500 from both sides of the equation, which gives us $x - 4500 = 3400 + 2000 - 6500$. Simplifying the right side of the equation, we get $x - 4500 = -1100$. Next, we add 4500 to both sides of the equation to isolate $x$, which gives us $x = -1100 + 4500$. Simplifying the right side of the equation, we get $x = 3400$.\n</think> <answer> $x = 3400$ </answer>",

        "scores": {

            "format_reward": 1.0,

            "answer_reward": 0.0,

            "reward": 0.0

        }

    }
```

(c) 错误主要是格式问题；其次图形相关的题目表现不是很好。  