"""练习 03 的参考实现。"""

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
        future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v

    def step(self, x_new: torch.Tensor, past_k: torch.Tensor | None, past_v: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_new, k_new, v_new = self.q(x_new), self.k(x_new), self.v(x_new)
        k_all = k_new if past_k is None else torch.cat([past_k, k_new], dim=1)
        v_all = v_new if past_v is None else torch.cat([past_v, v_new], dim=1)
        scores = q_new @ k_all.transpose(-2, -1) / math.sqrt(x_new.size(-1))
        output = torch.softmax(scores, dim=-1) @ v_all
        return output, k_all, v_all


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
    diff = (full - cached).abs().max().item()
    print("最大误差:", diff)
    assert torch.allclose(full, cached, atol=1e-6, rtol=1e-6)
