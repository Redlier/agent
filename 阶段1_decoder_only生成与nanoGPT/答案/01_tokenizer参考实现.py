"""练习 01 的参考实现；先独立补全，再逐项对照。"""


class CharTokenizer:
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
            raise ValueError(f"unknown characters: {unknown!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        unknown = [idx for idx in ids if idx not in self.itos]
        if unknown:
            raise ValueError(f"unknown ids: {unknown!r}")
        return "".join(self.itos[idx] for idx in ids)


def make_next_token_pair(ids: list[int], block_size: int) -> tuple[list[int], list[int]]:
    if block_size <= 0:
        raise ValueError("block_size 必须为正数")
    if len(ids) < 2:
        raise ValueError("至少需要两个 token 才能构造 next-token 样本")
    x = ids[:-1][:block_size]
    y = ids[1:][:block_size]
    return x, y


if __name__ == "__main__":
    tokenizer = CharTokenizer("abcabc")
    ids = tokenizer.encode("abcab")
    x, y = make_next_token_pair(ids, block_size=4)
    print("stoi:", tokenizer.stoi)
    print("vocab_size:", tokenizer.vocab_size)
    print("ids:", ids)
    print("x:", x)
    print("y:", y)
    assert tokenizer.decode(ids) == "abcab"
