"""练习：补全单头 attention 的 KV cache 更新。

目标不是追求速度，而是让 full 与 step 的输出逐位置一致。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CachedAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)

    def full(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(x.size(-1))
        t = x.size(1)
        future = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v

    def step(
        self,
        x_new: torch.Tensor,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO 1：只对 x_new 计算 q_new/k_new/v_new。
        # TODO 2：沿序列维拼接历史 K/V；第一次调用时直接使用新 K/V。
        # TODO 3：用 q_new 与全部 k 做 attention，再乘全部 v。
        raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    module = CachedAttention(4).eval()
    x = torch.randn(1, 5, 4)
    full = module.full(x)
    outputs, past_k, past_v = [], None, None
    with torch.no_grad():
        for t in range(x.size(1)):
            output, past_k, past_v = module.step(x[:, t : t + 1], past_k, past_v)
            outputs.append(output)
    cached = torch.cat(outputs, dim=1)
    print("最大误差:", (full - cached).abs().max().item())
