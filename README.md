# PyTorch Model Implementations

PyTorch を使って、代表的なニューラルネットワークと強化学習アルゴリズムを基礎から実装する学習用リポジトリです。

MLP、CNN、PPO、Transformer を題材に、モデルの構造だけでなく、データの準備、損失の計算、誤差逆伝播、パラメータ更新までの流れを理解することを目的としています。

> [!NOTE]
> このリポジトリは現在開発中です。コードには学習・検証途中の実装が含まれており、すべての `train.py` が最後まで動作する状態ではありません。

## 実装状況

| モデル | 内容 | 状況 |
| --- | --- | --- |
| MLP | 全結合ニューラルネットワーク | 準備中 |
| CNN | 畳み込みニューラルネットワーク | 準備中 |
| PPO | Actor-Critic、ロールアウト、GAE、Clipped Objective | 実装中 |
| Transformer | Causal Self-Attention、Multi-Head Attention、FFN、次トークン予測 | 実装中 |

## ディレクトリ構成

```text
.
├── README.md
└── src
    ├── MLP
    │   ├── MLP.py          # MLPモデル
    │   └── train.py        # MLPの学習処理
    ├── CNN
    │   ├── CNN.py          # CNNモデル
    │   └── train.py        # CNNの学習処理
    ├── PPO
    │   ├── PPO.py          # Actor-Criticモデル
    │   └── train.py        # PPOの学習処理
    └── Transformer
        ├── tranformer.py   # Transformerモデル
        └── train.py        # Transformerの学習処理
```

各モデルでは、ネットワーク定義と学習処理を別ファイルに分けています。

## 実行環境

- Python 3.12 以上
- PyTorch
- Gymnasium（PPOで使用）
- MuJoCo（`Hopper-v5` でPPOを実行する場合）

Python 3.12 以上を使用するのは、PPOの実装で `typing.override` を利用しているためです。

## セットアップ

仮想環境を作成してから、必要なライブラリをインストールします。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
```

PPOも試す場合は、GymnasiumとMuJoCoの追加依存関係をインストールします。

```bash
python -m pip install "gymnasium[mujoco]"
```

PyTorchのインストール方法は、OSやCUDAのバージョンによって異なります。GPUを利用する場合は、[PyTorch公式サイト](https://pytorch.org/get-started/locally/)で環境に合うコマンドを確認してください。

## 実装内容

### MLP

全結合層と活性化関数を組み合わせた、基本的なニューラルネットワークを実装する予定です。分類タスクを通して、順伝播、損失計算、誤差逆伝播の基本を確認します。

### CNN

畳み込み層、プーリング層、全結合層からなる画像分類モデルを実装する予定です。画像から局所的な特徴を抽出する仕組みを学びます。

### PPO

連続行動空間向けのActor-Criticモデルを実装しています。現在のコードには次の要素が含まれます。

- 正規分布からの行動サンプリング
- ロールアウトデータの収集
- Generalized Advantage Estimation（GAE）
- Clipped Surrogate Objective
- Value LossとEntropy Bonus
- ミニバッチによる複数エポック更新

検証環境にはGymnasiumの `Hopper-v5` を使用しています。

### Transformer

外部のTransformer実装に頼らず、主要な構成要素をPyTorchで組み立てています。

- Token EmbeddingとPosition Embedding
- Multi-Head Causal Self-Attention
- Layer NormalizationとResidual Connection
- Feed Forward Network
- 次トークン予測用の出力層

現在の学習コードでは、ランダムに生成したToken IDを使って処理の流れを確認しています。

## 今後の予定

- [ ] MLPモデルと学習ループの実装
- [ ] CNNモデルと学習ループの実装
- [ ] テストと実行例を追加
- [ ] 依存関係を `requirements.txt` または `pyproject.toml` に整理

## 補足

学習データやモデルの重みは容量が大きくなるため、Gitの管理対象外にしています。ローカルの `data/`、`datasets/`、および `*.pt`、`*.pth`、`*.ckpt` ファイルはコミットされません。
