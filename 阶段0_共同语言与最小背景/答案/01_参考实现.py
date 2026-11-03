"""练习 01 的一种参考实现。先独立补全练习，再对照此文件。"""

import torch
from torch import nn


def main() -> None:
    torch.manual_seed(7)
    x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    target = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0]])

    model = nn.Linear(in_features=1, out_features=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print("训练前预测：", model(x).detach().flatten().tolist())

    for step in range(101):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, target)
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
