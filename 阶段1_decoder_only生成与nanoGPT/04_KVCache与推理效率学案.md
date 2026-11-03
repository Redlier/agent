# 学案 04｜KV cache 与推理效率

## 1. 重复计算在哪里

不使用缓存时，生成每个新 token 都把完整上下文重新送入模型。第 `t` 步的注意力会重新计算历史位置的 K/V，即使历史 token 没变：

```text
第 1 步：计算 token 1 的 Q/K/V
第 2 步：重新计算 token 1、2 的 Q/K/V
第 3 步：重新计算 token 1、2、3 的 Q/K/V
```

这保证逻辑简单，但会重复工作。

## 2. 缓存的内容

对于每个 Transformer layer、每个 attention head，缓存历史 token 的 K 和 V：

```text
past_key_values[layer].key:   (B, nh, T_past, hs)
past_key_values[layer].value: (B, nh, T_past, hs)
```

当前新 token 只计算一个 `Q_new、K_new、V_new`，然后把 K/V 追加到缓存：

```text
K_all = concat(K_cache, K_new, dim=sequence)
V_all = concat(V_cache, V_new, dim=sequence)
Q_new 与 K_all 做注意力，再乘 V_all
```

当前步的 Q 形状是 `(B,nh,1,hs)`，所以输出只对应一个新位置。因为缓存中只有过去和当前，没有未来，增量步骤不再需要完整的上三角 mask；实现仍可能保留相应的边界处理。

## 3. 它节省什么、没有节省什么

KV cache 减少了历史 K/V 投影和重复的层前向计算，通常显著降低逐 token 解码延迟。它没有：

- 消除每个新 token 依赖前一个 token 的串行控制；
- 减少当前 query 与全部历史 key 的注意力读取；
- 免费获得更长上下文；
- 消除缓存占用的显存/内存。

缓存大小大致随 `层数 × batch × 头数 × 历史长度 × 每头维度 × 2(K/V)` 增长。长上下文和大 batch 时，缓存本身会成为重要资源。

## 4. 正确性对照

实现缓存时先做最重要的测试：同一组固定权重和输入，用 full recompute 与 incremental cache 计算每个位置的输出，检查最大绝对误差接近浮点误差。

```text
max_abs_diff = max(|full_output - cached_output|)
```

若差异很大，优先排查：

1. K/V 拼接维度；
2. 当前 token 的位置编号；
3. Q/K/V 的转置与 head 拆分；
4. dropout 是否在 eval 模式仍开启；
5. full 路径和 cached 路径是否使用了同一组参数。

## 5. prefill 与 decode

实际推理通常分两段：

- **prefill**：把用户已有 prompt 一次送入模型，建立整段 KV cache；
- **decode**：每次只输入一个新 token，读取并更新 cache。

prefill 仍可并行处理 prompt 的多个位置；decode 受自回归依赖限制。这个区分有助于理解“首 token 延迟”和“每 token 延迟”为什么不同。

## 6. 运行与观察

```powershell
& '.\.venv\Scripts\python.exe' '.\阶段1_decoder_only生成与nanoGPT\代码\04_kv_cache_compare.py'
```

程序用一个无 dropout 的单头注意力模块对比两条路径，并打印最大误差。它不是生产级 cache API，而是为了验证“缓存历史 K/V 与完整重算等价”这个原理。

## 7. 小练习

1. 如果 `B=2, nh=4, T_past=7, hs=8`，缓存的 K 形状是什么？当前 Q 的形状是什么？
2. 为什么不能只缓存 V 而完全不缓存 K？
3. 解释长上下文时 KV cache 的显存压力来自哪里。
4. 设计一个 cache 失效场景：如果改变了 prompt，却继续沿用旧 cache，会发生什么？
