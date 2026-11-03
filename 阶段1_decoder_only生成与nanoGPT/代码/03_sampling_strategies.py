"""阶段 1：在固定 logits 上比较 greedy、temperature、top-k 与 top-p。"""

from __future__ import annotations

import torch


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """返回过滤后的 1D logits；被过滤的类别设为负无穷。"""
    if logits.ndim != 1:
        raise ValueError("本教学函数只接受一维 logits")
    filtered = logits.clone()

    if top_k is not None:
        if not 1 <= top_k <= logits.numel():
            raise ValueError("top_k 必须在 1 到词表大小之间")
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered[filtered < threshold] = float("-inf")

    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p 必须在 (0, 1] 内")
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        # 至少保留累计概率达到阈值之前的第一个 token。
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits[remove] = float("-inf")
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(0, sorted_indices, sorted_logits)

    return filtered


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> tuple[int, torch.Tensor]:
    if temperature <= 0:
        return int(torch.argmax(logits)), torch.softmax(logits, dim=-1)
    filtered = filter_logits(logits / temperature, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sampled.item()), probs


def describe(name: str, logits: torch.Tensor, **kwargs: object) -> None:
    generator = torch.Generator().manual_seed(123)
    token_id, probs = sample_next(logits, generator=generator, **kwargs)
    kept = torch.where(probs > 0)[0].tolist()
    print(f"{name:12s} -> token={token_id}, kept={kept}, probs={probs.tolist()}")


def main() -> None:
    logits = torch.tensor([4.0, 3.2, 2.1, 0.5, -1.0])
    print("原始 logits:", logits.tolist())
    describe("greedy", logits, temperature=0.0)
    describe("temperature=.5", logits, temperature=0.5)
    describe("temperature=2", logits, temperature=2.0)
    describe("top_k=2", logits, temperature=1.0, top_k=2)
    describe("top_p=.75", logits, temperature=1.0, top_p=0.75)
    print("提示：一次随机样本不能代表整个采样分布，请固定种子并重复观察。")


if __name__ == "__main__":
    main()
