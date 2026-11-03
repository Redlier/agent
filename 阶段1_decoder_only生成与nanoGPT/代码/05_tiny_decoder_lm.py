"""阶段 1：在短文本上训练一个极小的字符级 decoder-only LM。

这是机制实验，不是有用的语言模型训练。默认 CPU、少量步数，便于阅读和复现。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (self.head_size**0.5)
        scores = scores.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class DecoderBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 2, n_head: int = 4, n_embd: int = 48) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"序列长度 {t} 超出 block_size={self.block_size}")
        positions = torch.arange(t, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # 合法起点为 0 到 N-block_size-1；torch.randint 的上界不包含，所以这里是 N-block_size。
    starts = torch.randint(0, data.numel() - block_size, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: TinyDecoderLM, train_data: torch.Tensor, val_data: torch.Tensor, block_size: int, batch_size: int, device: torch.device, batches: int = 5) -> tuple[float, float]:
    model.eval()
    values = []
    for data in (train_data, val_data):
        losses = []
        for _ in range(batches):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        values.append(sum(losses) / len(losses))
    return values[0], values[1]


def main() -> None:
    torch.manual_seed(1337)
    requested = os.environ.get("STAGE1_DEVICE", "cpu").lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("STAGE1_DEVICE=cuda 但当前 PyTorch 不可用 CUDA")
    device = torch.device(requested)

    root = Path(__file__).resolve().parents[1]
    text = (root / "数据" / "mini_corpus.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(ids))
    train_data, val_data = ids[:split], ids[split:]

    block_size, batch_size = 48, 16
    model = TinyDecoderLM(tokenizer.vocab_size, block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    steps = int(os.environ.get("STAGE1_STEPS", "100"))
    print(f"device={device}, vocab_size={tokenizer.vocab_size}, tokens={len(ids)}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}, steps={steps}")

    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 2, 1) == 0:
            train_loss, val_loss = estimate_loss(model, train_data, val_data, block_size, batch_size, device)
            print(f"step={step:3d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    prompt = "In "
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=120, temperature=0.8, top_k=8)
    print("prompt:", repr(prompt))
    print("sample:")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
