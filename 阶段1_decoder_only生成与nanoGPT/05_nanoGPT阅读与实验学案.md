# 学案 05｜nanoGPT 阅读路线与极小实验

## 1. 先知道仓库的角色

本目录的 `参考资料/nanoGPT` 是从官方 GitHub 仓库浅克隆的代码快照，记录在 [参考资料/阅读说明](参考资料/阅读说明.md)。它适合学习一个尽量短的 GPT 训练/采样实现，但仓库 README 已说明项目后来进入维护收尾状态；这里把它当作教学材料，不把它当作最新生产框架。

第一遍阅读不要从头到尾逐行翻译。先回答：

```text
数据从哪里来？
模型在哪里定义？
loss 在哪里算？
参数在哪里更新？
生成循环在哪里？
配置如何覆盖默认值？
```

## 2. 推荐阅读顺序

### 第一步：README 与配置

阅读 `README.md`、`config/train_shakespeare_char.py` 和 `configurator.py`，记录：

- 运行训练/采样需要哪些参数；
- 默认使用什么数据和模型规模；
- 命令行参数如何覆盖配置文件；
- 哪些步骤需要 GPU、编译扩展或数据下载。

### 第二步：数据准备

阅读 `data/shakespeare_char/prepare.py` 和同目录的训练配置。重点追踪：

```text
原始文本 -> 字符词表 -> encode -> train.bin/val.bin -> get_batch
```

确认 `x` 与 `y` 是否错开一位，以及 batch、block size 的形状。不要一开始研究 Shakespeare 数据本身。

### 第三步：模型前向

阅读 `model.py`，建议按以下符号定位：

1. `GPTConfig`：词表、block size、层数、头数、embedding 维度；
2. `CausalSelfAttention`：QKV、causal bias/flash attention、投影；
3. `MLP`：逐位置前馈层；
4. `Block`：LayerNorm、attention、MLP 和残差；
5. `GPT.forward`：token/position embedding、blocks、最终 logits 与 loss；
6. `GPT.generate`：取最后位置、温度、top-k、追加 token。

把每个模块的输入输出形状写在纸上。先不追究初始化系数和性能注释。

### 第四步：训练主循环

阅读 `train.py`，沿一次迭代追踪：

```text
get_batch -> model(X,Y) -> loss.backward()
-> grad clip/optimizer.step -> scheduler -> eval/checkpoint
```

注意混合精度、梯度累积、梯度裁剪和 checkpoint 都是工程包裹；核心 next-token loss 仍是阶段 1 的主线。

### 第五步：采样入口

阅读 `sample.py`，确认 checkpoint 如何加载、prompt 如何编码、`model.generate` 的输出如何解码。把采样参数和输出文本一起记录，不要只复制屏幕上的文本。

## 3. 代码对照问题

阅读本地仓库时，逐题写答案：

1. nanoGPT 的字符 tokenizer 如何保证 `encode/decode` 可逆？
2. `block_size` 同时限制了哪些张量维度和行为？
3. 为什么 `GPT.forward` 在有 targets 时返回 loss，没有 targets 时只返回 logits？
4. `generate` 为什么取 `logits[:, -1, :]`？
5. 代码中的 causal attention 有普通 mask 和 flash attention 两条路径时，二者要满足什么数值语义？
6. 哪些配置会改变参数量，哪些只影响采样行为？
7. 如果训练 loss 下降但生成全是同一个字符，先检查哪些数据、采样和上下文因素？

## 4. 本目录极小实验

运行：

```powershell
& '.\.venv\Scripts\python.exe' '.\阶段1_decoder_only生成与nanoGPT\代码\05_tiny_decoder_lm.py'
```

这个程序使用 `数据/mini_corpus.txt`、字符级 tokenizer 和很小的 decoder-only 模型。它的目的有三层：

1. 看见从数据切片到 loss 的真实训练闭环；
2. 看见训练后生成循环如何使用最后位置 logits；
3. 学会记录“模型规模、步数、提示、生成结果、限制”，而不是把一条样例当作结论。

默认配置优先保证可运行和可读。生成文本重复、语法不稳定或只记住训练语料，都属于小数据实验的正常边界。

建议一次只改一个变量：

- `block_size`：上下文范围；
- `n_layer/n_head/n_embd`：表示容量；
- `learning_rate` 或训练步数：优化过程；
- `temperature/top_k`：生成策略，不改变已训练参数。

每次实验保存配置和结果，写下你对差异的预期，再观察是否支持预期。

## 5. 边界与许可

本阶段不运行 nanoGPT 的大规模训练命令，也不自动下载其数据集或权重。仓库代码和文件的许可信息以本地 `参考资料/nanoGPT/LICENSE` 为准；论文 PDF 仅作为个人学习参考，引用时应回到其正式出版/预印本页面。
