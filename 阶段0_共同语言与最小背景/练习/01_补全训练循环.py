"""阶段 0 练习：补全一个 y = 2x + 1 的小型线性回归训练循环。

说明：本文件故意留有 5 个空位。请先阅读上级目录的 04_PyTorch入门学案.md。
补完 TODO 后运行：python 01_补全训练循环.py
参考实现位于 ../答案/01_参考实现.py；请先尝试，再查看答案。
"""

import torch
from torch import nn


def main() -> None:
    torch.manual_seed(7)

    # 数据形状均为 (样本数, 特征数)。这里每个输入只有 1 个特征。
    x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    target = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0]])

    # TODO 1：创建从 1 个输入特征映射到 1 个输出数值的线性模型。
    model = nn.Linear(in_features=1,out_features=1)

    # TODO 2：为数值回归创建均方误差损失函数。
    loss_fn = nn.MSELoss()

    # TODO 3：创建 SGD 优化器，传入 model.parameters()，学习率可设为 0.1。
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.1)

    print("训练前预测：", model(x).detach().flatten().tolist())

    for step in range(101):
        # TODO 4：在每次计算新梯度前清空旧梯度。
        optimizer.zero_grad()

        prediction = model(x)
        loss = loss_fn(prediction, target)

        # TODO 5：先反向传播，再让优化器更新参数。补两行。
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"step={step:3d}  loss={loss.item():.6f}")

    with torch.no_grad():
        final_prediction = model(x)
    print("训练后预测：", final_prediction.flatten().tolist())
    print("目标值：    ", target.flatten().tolist())
    print("学到的 weight:", model.weight.item())
    print("学到的 bias:  ", model.bias.item())


if __name__ == "__main__":
    main()
