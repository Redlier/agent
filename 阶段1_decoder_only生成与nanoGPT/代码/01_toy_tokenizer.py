"""阶段 1：用一个字符级 tokenizer 观察文本、ID 和 next-token 样本。"""

from __future__ import annotations


class CharTokenizer:
    """教学用字符 tokenizer；生产系统通常使用更复杂的子词方案。"""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - self.stoi.keys())
        if unknown:
            raise ValueError(f"文本含有词表之外的字符: {unknown!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        unknown = [idx for idx in ids if idx not in self.itos]
        if unknown:
            raise ValueError(f"ID 超出词表范围: {unknown!r}")
        return "".join(self.itos[idx] for idx in ids)


def main() -> None:
    text = "to be or not to be"
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)
    recovered = tokenizer.decode(ids)

    print("原文:", repr(text))
    print("词表大小:", tokenizer.vocab_size)
    print("字符到 ID:", tokenizer.stoi)
    print("编码:", ids)
    print("解码:", repr(recovered))
    assert recovered == text

    # next-token 训练样本：x 的每个位置预测 y 的对应位置。
    x = ids[:-1]
    y = ids[1:]
    print("x（上下文）:", x)
    print("y（下一字符目标）:", y)
    print("样本数:", len(x))
    print("第一个监督关系:", repr(tokenizer.decode([x[0]])), "->", repr(tokenizer.decode([y[0]])))

    try:
        tokenizer.encode("to?be")
    except ValueError as exc:
        print("未知字符示例:", exc)


if __name__ == "__main__":
    main()
