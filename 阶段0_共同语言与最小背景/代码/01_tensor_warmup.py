"""阶段 0 张量与梯度热身：只用 PyTorch 基础张量，不训练大型模型。"""

import torch


def show_tensor_basics() -> None:
    # Python 列表变成 PyTorch 张量；每个数显式写成浮点数。
    x = torch.tensor([1.0, 2.0, 3.0])
    print("x:", x)
    print("shape:", x.shape)  # 一维，长度为 3
    print("dtype:", x.dtype)

    # 对应位置相乘（不是矩阵乘法）。
    doubled = x * 2
    print("x * 2:", doubled)

    # 两个 2x2 张量做矩阵乘法。
    matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    vector = torch.tensor([[4.0], [7.0]])
    product = matrix @ vector
    print("matrix shape:", matrix.shape)
    print("vector shape:", vector.shape)
    print("matrix @ vector:", product)
    print("product shape:", product.shape)


def show_autograd() -> None:
    # w 是一个可学习参数；x 是本次输入，不需要更新。
    w = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(2.0)
    target = torch.tensor(5.0)

    prediction = w * x
    loss = (prediction - target) ** 2
    print("\nprediction:", prediction.item())
    print("loss:", loss.item())

    # 反向传播计算 d(loss)/d(w)，此行本身不会改变 w。
    loss.backward()
    print("gradient d(loss)/d(w):", w.grad.item())
    print("w after backward (unchanged):", w.item())

    # 只演示一次最简单的梯度下降更新，方便观察“算梯度”和“改参数”是两步。
    learning_rate = 0.1
    with torch.no_grad():
        w -= learning_rate * w.grad
    print("w after one manual update:", w.item())


if __name__ == "__main__":
    torch.manual_seed(0)
    show_tensor_basics()
    show_autograd()
