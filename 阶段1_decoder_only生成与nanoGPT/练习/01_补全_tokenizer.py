"""练习：补全一个最小字符 tokenizer 和 next-token 样本构造。

先阅读 01_文本与Tokenizer学案.md。故意保留关键空位，不要先看答案。
"""


class CharTokenizer:
    def __init__(self, text: str) -> None:
        # TODO 1：建立稳定的字符词表；注意同一字符只能有一个 ID。
        self.stoi = None
        self.itos = None

    @property
    def vocab_size(self) -> int:
        # TODO 2：返回词表大小。
        return 0

    def encode(self, text: str) -> list[int]:
        # TODO 3：把每个字符转换成整数 ID，并对未知字符给出清晰错误。
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        # TODO 4：把 ID 转回字符；不要默默忽略非法 ID。
        raise NotImplementedError


def make_next_token_pair(ids: list[int], block_size: int) -> tuple[list[int], list[int]]:
    # TODO 5：返回错开一位的 x 和 y，并截取最多 block_size 个位置。
    raise NotImplementedError


if __name__ == "__main__":
    tokenizer = CharTokenizer("abcabc")
    ids = tokenizer.encode("abcab")
    x, y = make_next_token_pair(ids, block_size=4)
    print(tokenizer.stoi, tokenizer.vocab_size, ids, x, y)
    assert tokenizer.decode(ids) == "abcab"
