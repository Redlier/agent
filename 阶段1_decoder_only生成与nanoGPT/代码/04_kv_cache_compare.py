"""阶段 1：比较完整重算与增量 KV cache 的单头注意力输出。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TinyAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """一次计算完整序列，输入/输出形状都是 (B,T,C)。"""
        _, t, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(x.size(-1))
        future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.out_proj(weights @ v)

    def forward_step(
        self,
        x_new: torch.Tensor,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """只处理一个新位置，并返回更新后的 K/V。"""
        q_new = self.q_proj(x_new)
        k_new = self.k_proj(x_new)
        v_new = self.v_proj(x_new)
        k_all = k_new if past_k is None else torch.cat([past_k, k_new], dim=1)
        v_all = v_new if past_v is None else torch.cat([past_v, v_new], dim=1)
        scores = q_new @ k_all.transpose(-2, -1) / math.sqrt(x_new.size(-1))
        weights = torch.softmax(scores, dim=-1)
        output = self.out_proj(weights @ v_all)
        return output, k_all, v_all


def main() -> None:
    torch.manual_seed(11)
    attention = TinyAttention(channels=8).eval()
    x = torch.randn(1, 6, 8)

    with torch.no_grad():
        full = attention.forward_full(x)
        pieces: list[torch.Tensor] = []
        past_k = past_v = None
        for position in range(x.size(1)):
            output, past_k, past_v = attention.forward_step(
                x[:, position : position + 1], past_k, past_v
            )
            pieces.append(output)
        cached = torch.cat(pieces, dim=1)

    max_diff = (full - cached).abs().max().item()
    print("完整重算输出形状:", tuple(full.shape))
    print("最终缓存 K 形状:", tuple(past_k.shape))
    print("最终缓存 V 形状:", tuple(past_v.shape))
    print("full vs cache 最大绝对误差:", max_diff)
    assert torch.allclose(full, cached, atol=1e-6, rtol=1e-6)
    print("KV cache 数值等价检查: PASS")


if __name__ == "__main__":
    main()
