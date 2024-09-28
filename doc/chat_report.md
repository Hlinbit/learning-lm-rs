# 聊天机器人

为了使推理引擎支持更多数据的模型推理，更加良好的的使用体验，我们主要做了如下的功能开发，包括：
- 对float16和float32的模型推理的支持。
- 命令行形式的单机的用户交互。
- 针对不同用户的数据管理。
- 基本的llama模型的推理计算。

## 基础功能

chat模型与story模型都是按照llama模型为基本结构的具体呈现。无论是结构还是计算流程几乎完全相同，意味着计算的story模型的代码可以直接计算chat模型。

但是有一处不同的地方：story模型的`tie_word_embeddings`为`true`嵌入层参数（embed_token）和最后的输出层（lm_head）是完全相同的，但是chat模型这两部分是两组不同的参数。

通过chat模型的配置信息和meta信息，我们可以对chat模型有一个基本了解：
- 模型参数类型为32位浮点数
- 模型的中间状态长度为312
- 模型的中间层数为10
- 模型的词典规模是32002



## 用户交互设计

推理引擎支持简单的用户交互。启动推理引擎之后，会展示用户交互命令行。交互命令行支持以下特殊命令：

- `login`，用于用户注册，输入用户名之后，推理引擎会查看是否有用户的历史推理记录，如果存在则加载历史记忆，用于之后推理。
- `logout`，用于切换用户，并存档用户推理记录。推理引擎会在用户登出时，对历史记忆进行存档，方便用户基于记忆继续推理。
- `exit` & `quit`，用于退出推理引擎。
- 除了以上的特定命令，用户输入的任何内容都会用于模型推理。

### 用户记忆数据管理

用户交互的核心在于用户的信息管理。在采用Transformer架构的模型中，历史计算的K向量和V向量被视为模型的“记忆”。对于每个用户来说，他们与模型交流的内容不同，记忆也不同，而模型通过记忆中包含的上下文信息给出特定的的推理结果。因此如何管理每个用户的记忆，就是用户交互的核心工作之一。

在本文的工作中，用户数据管理主要包含两部分工作。
- 我们通过将记忆落盘到文件中，持久化用户的记忆。为此，我们设计了一种特殊的文件格式来保存用户的历史推理数据。
- 我们通过对K向量和V向量缓存采用块淘汰的策略，处理历史内容超出模型承载能力的情况。

#### 记忆数据持久化

为了支持推理引擎的用户切换，推理引擎必须能够持久化保存用户数据，同时支持用户数据的恢复。为了高效存储用户数据，本文设计了一种文件格式，如下图所示：

<div style="text-align: center;">
  <img src="./picture/persistence.svg" alt="self-attention" width="" height="">
</div>

在实际情况中，K向量和V向量分别存放在两个文件中，每个文件内容是：

- 首先文件最开始8个字节为按照`usize`格式保存的历史数据的长度，也就是K向量和V向量的个数。
- 之后是按照层存放的，每层的K向量或者V向量，每个向量包含两部分：
  - 第一部分是元信息。'T'字节后面跟着数据类型，包括`f16`和`f32`两种类型，'L'字节后面紧跟着向量数据包含的字节数。
  - 第二部分是数据。持久化数据时，会将数据转化为字节，存放在data区，读取数据时，按照数据格式读取这些字节，恢复出原始数据。
  
这些向量在文件中按照途中所示的格式紧密排列。
  

#### 历史数据溢出的处理---块淘汰策略

随着用户的交流信息增多，将会逐渐达到模型的承载能力（max_position_embeddings）。因此需要时不时清理掉过早的信息为新生成的token提供新的空间，在推理引擎中，我们采用如图所示的策略：

<div style="text-align: center;">
  <img src="./picture/block.svg" alt="self-attention" width="" height="">
</div>

随着历史向量逐渐积累，缓存中存放的K（或者V）向量的个数逐渐达到了max_position_embeddings。此时，我么会淘汰掉最前面一块数据（长度为128），并将后面的数据移动到缓存最前，为新生成的K向量和V向量腾出空间。

## 支持16位浮点数推理计算

参数量化是常见且有效的降低模型资源占用的技术。量化通过减少模型权重和激活的精度来降低模型的大小和计算复杂度。

例如：16位浮点数相比于32位浮点数，可以节省模型计算时一半的内存占用；同时站在运算器的角度来看，16位数据的计算量是32位数据的计算量的一般，计算速度也可以相应提高（然而由于大部分CPU不支持16位浮点数计算，需要先将其转化为32位浮点数进行计算，造成很多额外的开销）。

因此，为了让推理引擎支持更低精度的数据类型，进而实现更少的推理资源占用和更快的运算速度，我们决定进一步开发推理引擎，使其支持16位浮点数计算。

### 需求梳理和实现

已有代码中已经定义了模型相关的泛型结构体`Llama<T>`和`LLamaParams<T>`，以及基础的张量泛型结构体`Tensor<T>`。然而与推理相关的函数（例如`forward`）,以及张量计算的算子都是基于32位浮点数实现的。

结合rust语言的特性，实现支持16位浮点数运算的同时保持原有的计算流程，成本最低的实现方式就是将`Llama<T>`的成员函数以及推理所需的算子函数都改为泛型函数。

在rust中定义泛型函数，我们需要对泛型参数`T`做出严格的描述。通过总结推理过程中关于`Tensor<T>`的运算，我们得到泛型参数`T`需要满足以下特性：

- 支持相同类型的四则运算。
- 支持科学计算，例如指数运算，平方根运算等。
- 支持与rust中基础类型（如`usize`）进行相互转换。

根据目前需求，泛型参数`T`包括32位浮点数`f32`和16位浮点数`f16`，`f32`作为基础类型自然满足以上特性。

然而`f16`并不是rust中原生支持的类型。不过已经有第三方库`half`支持f16的相关运算。在引入`half`库的`num-traits`特性之后，`half::f16`可以完全支持上面的全部特性。

### 性能分析

推理引擎的计算过程大致可以分为三部分：K，Q，V投影，Self-Attention的计算以及MLP计算。

为了保证实验的代表性，我们按照release标准编译程序，计算这三部分的延迟。下表展示了统计结果：

| 阶段     | 输入token长度   | 计算耗时   | 数据类型 |
| -------- | --------- | ------- | -------- |
| QKV projection | 17   | 10.18 ms     |  float16|
| Self Attention | 17    | 6.94 ms        |float16|
| MLP   | 17    | 65.42ms     |float16 |
| Layer cost   | 17    | 82.61ms     |float16 |
| QKV projection | 17   | 1.58ms   |float32|
| Self Attention | 17    | 1.47ms   |float32|
| MLP   | 17    | 11.64ms     |float32 |
| Layer cost   | 17    | 14.32ms     |float32 |

从实际测试结果来看：数据类型为float16模型的推理速度要显著低于float32数据类型的模型。其原因在于`half::16`并不在语言层面支持科学计算，因此在`half`中采用了先将`f16`转为`f32`然后再进行计算的策略。从`half`中对`sqrt`的实现，可见一斑：

```rust
#[inline]
fn sqrt(self) -> Self {
    Self::from_f32(self.to_f32().sqrt())
}
```

由于f32和f16两种数据的内部二进制排列标准完全不同，必然涉及到额外的，相当大的额外计算和内存分配工作。因此f16模型推理相较于f32有较大下降也是可以理解的。

### 资源分析

分别让float16模型和float32模型推理根据相同输入的prompt，进行一次完整的推理，分别统计推理引擎对应进程的内存占用，得到以下实验结果。可见：
- 模型存储角度上看，float16模型缩小至相同的float32模型的一半。
- 从推理资源占用角度来看，float16模型的内存占用也大概是float32模型的一半。

可见通过支持float16数据类型的模型推理，确实实现了推理资源占用的减少。

| 数据类型     | 物理内存占用(RSS)  | 模型大小  |
| -------- | --------- | ------- |
| float16 | 153.1M   | 63M  | 
| float32 | 282.8M    | 125M     |

## 精度分析

模型推理的关键在于保证推理精度，也就是保证推理结果的合理。为了验证float16精度推理是否正确，精度是否符合预期，本文采用两种方法进行验证：

- 中间数据对比
- 推理结果对比

为了检测模型精度，我们使用pytorch将原有的chat模型进行参数量化，生成数据相同，但类型为float16的模型：chat16。


### 中间数值对比

实验采用每层layer产出的隐藏状态作为对比对象，参照对象是higgingface开发的`transformer`框架依靠chat16计算产生的每一层隐藏状态。

在对比计算结果时，必须设置合理的误差范围，而不能直接比较值是否相同。原因在于：

- 二进制难以精确表达小数，float16由于自身限制，其本身表达精度为0.001。
- `Transformer`框架并不支持在CPU上的float16计算，会将数据自动转化为float32进行计算。而我们的推理引擎，需要再计算过程中反复将数据再f16和f32之间来回转换，相应的精度损失较大。

由于以上两种造成偏差的原因，我们设置误差范围为0.04。通过比较一次推理中，每一层输出的结果与`transformer`框架的每一层计算结果之后，发现，全部元素的绝对值误差小于设定的误差范围，可以得出结论：推理引擎的计算误差基本符合预期。

对比脚本可见：[对比脚本](../pytorch_imp/compare_chat.py)

### 推理结果对比

使用chat16模型和相同的prompt，对比`Transformer`提供的`pipeline`推理结果([pipeline脚本](../pytorch_imp/chat.py))和我们的推理引擎的推理结果，得到的结果如下：

- pipeline 推理结果
```
<|im_start|>user
What are some potential applications for quantum computing?<|im_end|>
<|im_start|>assistant
1. Quantum computing: Quantum computing is a popular choice for quantum computing because it allows for the creation of new quantum states, which can be used to generate new quantum states.

2. Quantum computing: Quantum computing is a popular choice for quantum computing because it allows for the creation of new quantum states that can be used to solve problems.

3. Quantum computing: Quantum computing is a popular choice for quantum computing because it allows for the creation of new quantum states that can be used to solve problems.

4. Quantum computing: Quantum computing is a popular choice for quantum computing because it allows for the creation of new quantum states that can be used to solve problems.

5. Quantum computing: Quantum computing is a popular choice for quantum computing because it allows for the creation of new quantum states that can be used to solve problems.

6. Quantum computing: Quantum computing is a popular choice for quantum computing because it allows for the creation of new quantum states that can be used to solve problems.

7. Quantum computing: Quantum computing is a popular choice for quantum computing because it allows for the creation of new quantum states that can be used to solve problems.
```

推理引擎推理结果：

```
Quantum computing is a rapidly growing field of research that has been successfully implemented in recent years. Here are some potential applications for quantum computing:

1. Quantum computing: Quantum computing is a promising field of study that has been successful in improving the performance of quantum computing. Quantum computing has been used to develop new applications, including quantum computing, which can be used to simulate quantum states.

2. Quantum computing: Quantum computing is a promising field of study that has been successful in improving the performance of quantum computing. Quantum computing has been used to develop new applications in various fields, including medicine, medicine, and medicine.

3. Quantum computing: Quantum computing has been used to develop new applications in various fields, including medicine, medicine, and medicine. Quantum computing has been used to develop new applications in various fields, including medicine, medicine, and medicine.

4. Quantum computing: Quantum computing has been used to develop new applications in various fields, including medicine, medicine, and medicine. Quantum computing has been used to develop new applications in various fields, including medicine, medicine, and medicine.

5. Quantum computing: Quantum computing has been used to develop new applications
```

可以看出，虽然结果不同，但是内容保持了相当的合理性，由于不支持设置重复惩罚，因此和pipeline产生了一样的回答重复的问题，但这与模型精度无关。

### 小结

通过完整的实验，我们验证了推理引擎在对float16模型的推理过程中保持了预期的精度。支持float16数据类型推理，有效的扩展了推理引擎的支持范围，针对低精度数据进行推理在保证预期精度的同事，有效的减少了资源的占用，尽管由于rust没有完整支持float16类型的计算，造成了性能损失较大，但是随着语言的不断演进，成熟完备的解决方案将会出现，推理引擎将会具备资源占用减少和性能提升的双重优势。