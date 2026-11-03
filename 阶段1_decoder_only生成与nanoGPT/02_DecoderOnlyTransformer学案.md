# 学案 02｜Decoder-only Transformer 的数据流

## 1. 从概率分解开始

给定 token 序列 `x_1, ..., x_T`，自回归语言模型把联合概率写成：

```text
P(x_1, ..., x_T) = Π_t P(x_t | x_1, ..., x_{t-1})
```

训练时所有位置可以并行计算，但每个位置只能读取自己及左侧 token。推理时模型每次只产生一个新 token，再把它放回上下文。

模型最后输出的不是文字，而是每个位置对整个词表的**未归一化分数 logits**：

```text
logits: (B, T, V)
```

对最后一维做 softmax 才得到概率。训练交叉熵内部通常已经包含稳定的 log-softmax，因此不要先手动 softmax 再传给 `CrossEntropyLoss`。

## 2. Decoder-only block 的全局结构

一个简化的 decoder-only Transformer 可以画成：

```text
token IDs (B,T)
   │ embedding + position
   ▼
x (B,T,C)
   │
   ├─ LayerNorm -> causal self-attention -> residual add
   │
   └─ LayerNorm -> MLP/FFN             -> residual add
          （重复 N 层）
   ▼
final LayerNorm -> Linear(C,V)
   ▼
logits (B,T,V)
```

不同实现可能使用 post-LN、RoPE、RMSNorm、权重绑定或不同位置表示。先掌握数据流，不要把某个实现细节当成 decoder-only 的唯一形式。

## 3. Embedding 与位置

Token embedding 是形状为 `(V,C)` 的参数表。输入 ID `(B,T)` 经过查表得到 `(B,T,C)`。同一个 token 在不同位置需要有位置线索，否则注意力看到的是一个无序集合。nanoGPT 的早期实现使用可学习的位置 embedding；其他现代模型可能使用 RoPE 等相对位置方法。

位置表示不是“把位置编号直接加到文本上”，而是把位置编码成与 token 表示同维度的向量，再参与后续计算。

## 4. 单头因果自注意力

设 `x` 的形状为 `(B,T,C)`，线性投影产生：

```text
Q, K, V: (B, nh, T, hs)
```

其中 `nh` 是头数，`hs=C/nh`。注意力分数和加权结果为：

```text
scores = Q K^T / sqrt(hs)       -> (B, nh, T, T)
weights = softmax(scores, dim=-1)
y = weights V                   -> (B, nh, T, hs)
```

`scores[t, s]` 表示位置 `t` 对位置 `s` 的关注程度。因果 mask 把 `s>t` 的未来位置设为负无穷，使 softmax 后权重为 0：

```text
允许：□
       □ □
       □ □ □
       □ □ □ □
```

注意力不是“每个 token 只看前一个 token”；它可以读取全部历史，只是不能读取未来。

## 5. 多头、残差与 MLP

多头注意力把通道维拆成多个头，每个头独立计算 Q/K/V，再拼回 `(B,T,C)`。不同头可以学习不同关系，但“每个头对应一个固定语义”不是保证。

MLP 通常是逐位置的两层前馈网络：

```text
(B,T,C) -> Linear(C,4C) -> 激活 -> Linear(4C,C) -> (B,T,C)
```

它混合同一位置的通道信息；注意力负责跨位置通信。残差连接把子层输入加回输出，帮助深层网络优化；LayerNorm/RMSNorm 控制表示尺度。

## 6. 为什么训练能并行、生成仍逐步

训练样本的完整目标序列已知，因果 mask 一次就能让每个位置计算自己的条件分布，因此可并行计算 `T` 个位置。生成时未来 token 未知，第 `t+1` 个输入要等第 `t` 个 token 采样出来，所以控制流程仍是循环。

这一区别很重要：**并行训练不等于并行自回归生成**。KV cache 只能减少每一步重复的历史 K/V 计算，不能消除 token 之间的依赖。

## 7. 形状排查表

| 对象 | 常见形状 | 含义 |
|---|---|---|
| token IDs | `(B,T)` | 整数类别编号 |
| embedding | `(B,T,C)` | 每个 token 的连续表示 |
| Q/K/V（拆头后） | `(B,nh,T,hs)` | 每头的查询/键/值 |
| attention scores | `(B,nh,T,T)` | 位置两两分数 |
| block 输出 | `(B,T,C)` | 通道维回拼后的表示 |
| logits | `(B,T,V)` | 每个位置对词表的分数 |
| 目标 | `(B,T)` | 下一 token 的整数 ID |

## 8. 运行与观察

```powershell
& '.\.venv\Scripts\python.exe' '.\阶段1_decoder_only生成与nanoGPT\代码\02_causal_attention.py'
```

程序会打印一个小注意力矩阵，并检查未来位置权重是否为 0。修改 mask 或把 `-inf` 换成 0，再观察为什么因果约束会失效。

## 9. 常见误解

- decoder-only 不是“只有 decoder 的最后一半”；它仍有完整的 self-attention、MLP、残差和归一化堆叠。
- causal mask 是注意力分数上的信息约束，不是把未来 token 从内存中删除。
- logits 不是概率，也不是已经选好的 token。
- `Linear(C,V)` 的 `V` 必须与 tokenizer 词表大小匹配。
- 注意力权重可视化能帮助观察计算，但不能单独证明模型“在解释什么”。

## 10. 小练习

1. 给定 `B=2, T=5, C=16, nh=4`，写出 `hs` 和 Q/K/V 的形状。
2. 为什么第 0 行的 causal attention 只能看到第 0 列？
3. 如果把 mask 的未来位置设成 0 而不是 `-inf`，softmax 后会发生什么？
4. 解释为什么 MLP 不需要 `T×T` 的矩阵，而 attention 需要。
