"""练习：补全一个只依赖 next_logits 函数的自回归生成循环。"""

from __future__ import annotations

import torch


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """输入形状 (B,V)，输出形状 (B,1)。"""
    # TODO 1：处理 temperature；确认除法发生在 softmax 之前。
    scaled = logits
    # TODO 2：如果设置 top_k，把其他 logit 设为 -inf，至少保留一个类别。
    filtered = scaled
    # TODO 3：softmax 后用 torch.multinomial 采样。
    raise NotImplementedError


def generate(
    idx: torch.Tensor,
    next_logits,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    stop_id: int | None = None,
) -> torch.Tensor:
    """next_logits(context) 应返回形状 (B,V) 的最后位置 logits。"""
    # TODO 4：循环 max_new_tokens 次；每轮只把最后 block_size 个 token 作为上下文。
    # TODO 5：采样一个 (B,1) token，拼接到 idx，并在 stop_id 后停止。
    raise NotImplementedError


if __name__ == "__main__":
    # 这个假模型总是偏好 (last_id + 1) % vocab_size，便于检查循环形状。
    vocab_size = 5

    def fake_next_logits(context: torch.Tensor) -> torch.Tensor:
        logits = torch.full((context.size(0), vocab_size), -10.0)
        next_id = (context[:, -1] + 1) % vocab_size
        logits.scatter_(1, next_id[:, None], 10.0)
        return logits

    start = torch.tensor([[0, 1]])
    result = generate(start, fake_next_logits, max_new_tokens=3, block_size=4, temperature=0.0)
    print(result)
