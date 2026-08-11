import torch
import torch.nn as nn
import torch.optim as optim
from Transformer.tranformer import SmallTransformer

tokens = torch.randint(0, 10000, (8, 32))
y = torch.randint(0, 10000, (8, 32))
net = SmallTransformer()
optimizer = optim.Adam()
loss_fn = nn.CrossEntropyLoss()

for iter in range(1000):
    # Transformerで次トークンのlogitsを計算する。
    logits = net(tokens)

    # CrossEntropyLoss用に、バッチ次元と系列次元をまとめる。
    logits = logits.reshape(-1, 10000)

    # 元の教師データyは変更せず、Loss計算用の1次元Tensorを作る。
    targets = y.reshape(-1)

    # 予測と正解Token IDからLossを計算する。
    # targetに対応するTokenの確率に-log()をした値
    loss = loss_fn(logits, targets)

    # 前回のiterationで計算された勾配を消去する。
    optimizer.zero_grad()

    # 誤差逆伝播によって各パラメータの勾配を計算する。
    loss.backward()

    # 勾配を使ってパラメータを更新する。
    optimizer.step()

    # 最後のトークンのロジットのみを抽出する
    next_token_logits = logits[:, -1, :]

    # 各語彙のlogitを確率に変換する。
    # shape: (B, vocab_size) -> (B, vocab_size)
    probs = torch.softmax(next_token_logits, dim=-1)

    # 確率分布 probs に従って、各バッチから1つのToken IDをサンプリングする。
    # probs.shape = (B, vocab_size)
    next_token_id = torch.multinomial(probs, num_samples=1)

    # 次トークンに系列長方向の次元を1つ追加する。
    # (B,) -> (B, 1)
    next_token = next_token_id.unsqueeze(-1)

    # 既存のToken列の末尾に、生成した1Tokenを追加する。
    # (B, T) + (B, 1) -> (B, T+1)
    tokens = torch.cat([tokens, next_token], dim=1)
