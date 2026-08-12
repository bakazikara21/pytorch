import torch.nn as nn
from torch import Tensor

class CNN(nn.Module):
    # 32 * 32の画像を入力したときに、画像分類するnetwork
    def __init__(self, num_classes: int = 10, hidden_size: int = 128) -> None:
        # num_classes: 多クラス分類のクラス数
        # hidden_size: 分類器に対応するMLPの隠れ層のノード数
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32×32 -> 16×16
            nn.Conv2d(
                in_channels=3,      # 入力は(R,G,B)の3チャネル
                out_channels=32,    # 出力チャネルを32にする
                kernel_size=3,      # カーネルは 3*3 の行列にする
                padding=1,          # paddingは縦横1マスずつ->縦と横が不変になる
            ),
            nn.BatchNorm2d(num_features=32),        # 32個の各チャネルでバッチ正規化を行う
            nn.ReLU(inplace=True),                  # 値をその場で書き換える（コピーをしない）
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            # 2*2の最大プーリングにより縦横のサイズが1/2になる
            nn.Dropout2d(0.2),                      # 20%のノードの出力を0にする

            # Block 2: 16×16 -> 8×8
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3),

            # Block 3: 8×8 -> 4×4
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.4)
        )

        self.flatten = nn.Flatten()                 # 畳み込み層から全結合層へ入力するために1次元ベクトル化

        self.classifier = nn.Sequential(
            nn.Linear(4*4*128,hidden_size),         # (32/8)pixel * (32/8)pixel * 128チャネル
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_size,num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

'''
入力
[B, 3, 32, 32]

↓ Conv
[B, 32, 32, 32]

↓ Conv
[B, 32, 32, 32]

↓ MaxPool
[B, 32, 16, 16]

↓ Conv
[B, 64, 16, 16]

↓ Conv
[B, 64, 16, 16]

↓ MaxPool
[B, 64, 8, 8]

↓ Conv
[B, 128, 8, 8]

↓ Conv
[B, 128, 8, 8]

↓ MaxPool
[B, 128, 4, 4]

↓ Flatten
[B, 4*4*128]

↓ Linear
[B, hidden_size]

↓ Linear
[B, num_classes]

↓
出力
'''