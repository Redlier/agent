"""阶段 1：手写带 causal mask 的 scaled dot-product attention。"""

import math

import torch


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """输入形状为 (B, heads, T, head_dim)，返回输出与注意力权重。"""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    t = q.size(-2)
    future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def main() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, 1, 4, 3)
    k = torch.randn(1, 1, 4, 3)
    v = torch.arange(1.0, 13.0).reshape(1, 1, 4, 3)
    output, weights = causal_attention(q, k, v)

    matrix = weights[0, 0]
    print("attention 权重矩阵（行是 query 位置，列是 key 位置）:")
    print(matrix)
    print("输出形状:", tuple(output.shape))
    print("未来位置最大权重:", matrix.triu(diagonal=1).abs().max().item())

    # 这个断言是本程序最重要的验收：当前位置不能读到未来。
    assert torch.allclose(matrix.triu(diagonal=1), torch.zeros_like(matrix))
    assert torch.allclose(matrix.sum(dim=-1), torch.ones(4))
    print("causal mask 检查: PASS")


if __name__ == "__main__":
    main()
