"""练习 02 的参考实现。"""

from __future__ import annotations

import torch


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / temperature
    if top_k is not None:
        top_k = min(max(int(top_k), 1), scaled.size(-1))
        values, _ = torch.topk(scaled, top_k, dim=-1)
        scaled = scaled.masked_fill(scaled < values[:, [-1]], float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(idx: torch.Tensor, next_logits, max_new_tokens: int, block_size: int, temperature: float = 1.0, top_k: int | None = None, stop_id: int | None = None) -> torch.Tensor:
    if idx.ndim != 2:
        raise ValueError("idx 必须是 (B,T) 整数张量")
    for _ in range(max_new_tokens):
        context = idx[:, -block_size:]
        logits = next_logits(context)
        next_id = sample_from_logits(logits, temperature=temperature, top_k=top_k)
        idx = torch.cat([idx, next_id], dim=1)
        if stop_id is not None and bool(torch.all(next_id == stop_id)):
            break
    return idx


if __name__ == "__main__":
    vocab_size = 5

    def fake_next_logits(context: torch.Tensor) -> torch.Tensor:
        logits = torch.full((context.size(0), vocab_size), -10.0)
        next_id = (context[:, -1] + 1) % vocab_size
        logits.scatter_(1, next_id[:, None], 10.0)
        return logits

    result = generate(torch.tensor([[0, 1]]), fake_next_logits, 3, 4, temperature=0.0)
    print(result.tolist())
