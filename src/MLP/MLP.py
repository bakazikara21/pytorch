import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):
    def __init__(self, input_size: int = 28*28, output_size: int = 10, hidden_size: int = 128) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            # Block 1: input_size -> hidden_size
            nn.Linear(input_size,hidden_size),  # 入力に重みを掛けてバイアスを加える全結合層。重みWとバイアスbを学習する
            nn.ReLU(inplace=True),              # 値をその場で書き換える（コピーをしない）

            # Block 2: hidden_size -> hidden_size
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(inplace=True),

            # Output: hidden_size -> output_size
            nn.Linear(hidden_size,output_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        return x

'''
nn.Linear(784, 128)
ならば、
入力 X   : [batch_size, 784]
重み W   : [128, 784]
バイアスb : [128]
となります。したがって、
X @ W.T + b
という計算を行っています。
'''