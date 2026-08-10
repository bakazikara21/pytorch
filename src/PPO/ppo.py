from typing import override
import torch.nn as nn
import torch
from torch import Tensor
from torch.distributions import Normal


class PPO(nn.Module):
    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        # Actorネットワーク
        self.actor: nn.Sequential = nn.Sequential(
            # 状態ベクトルをhidden_dim次元へ変換します。
            nn.Linear(state_dim, hidden_dim),

            # 非線形変換を加えます。
            nn.Tanh(),

            # hidden_dim次元の特徴量をさらに変換します。
            nn.Linear(hidden_dim, hidden_dim),

            # 再び非線形変換を加えます。
            nn.Tanh(),

            # 各行動次元に対応する平均μを出力します。
            nn.Linear(hidden_dim, action_dim),
        )

        # Criticネットワーク
        self.critic: nn.Sequential = nn.Sequential(
            # 状態ベクトルを hidden_dim 次元へ変換します。
            nn.Linear(state_dim, hidden_dim),

            # 非線形変換を加えます。
            nn.Tanh(),

            # hidden_dim 次元の特徴量をさらに変換します。
            nn.Linear(hidden_dim, hidden_dim),

            # 再び非線形変換を加えます。
            nn.Tanh(),

            # 状態価値 V(s) に対応するスカラーを出力します。
            nn.Linear(hidden_dim, 1),
        )

        # log(標準偏差)を学習可能なパラメータとして定義します。
        self.log_std: nn.Parameter = nn.Parameter(
            torch.zeros(action_dim)
        )

    def forward(self, state: Tensor) -> tuple[Normal, Tensor]:
        # Actorから各行動次元の平均を計算します。
        mu: Tensor = self.actor(state)

        # 学習可能なlog標準偏差から、正の標準偏差を作ります。
        std: Tensor = torch.exp(self.log_std)

        # 平均と標準偏差から連続行動の正規分布を作ります。
        dist: Normal = Normal(mu, std)

        # Criticから状態価値を計算します。
        value: Tensor = self.critic(state)

        return dist, value

    @override
    def __call__(self, state: Tensor) -> tuple[Normal, Tensor]:
        return super().__call__(state)
