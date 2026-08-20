from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import SmallTransformer
from transformer import EncoderDecoder

# GPUを使用する
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 学習する文字列
y = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"

# 学習対象の文字列から、教師データ、入力データ用のToken IDの配列を作る。
encoder_decoder = EncoderDecoder(y)
y_tokens = encoder_decoder.encode(y)

# Vocabularyのサイズを取得する
vocab_size = encoder_decoder.get_vocab_size()

# 小型Transformerモデルを生成
net = SmallTransformer(vocab_size).to(device)

# 最適化アルゴリズムと損失関数を設定
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()
epochs = 1000

# Transformerへの入力は教師データを一文字ずらした文字列とする
x_tokens = encoder_decoder.encode(' ' + y[:-1]).unsqueeze(0).to(device)

# logits=(33, 118) に対して targets=(33,)が必要
targets = y_tokens.reshape(-1).to(device)

for epoch in range(epochs):
    # Transformerで次トークンのlogitsを計算する。
    logits = net(x_tokens)

    # 最後のトークンのロジットのみを抽出する
    next_token_logits = logits[:, -1, :]

    # CrossEntropyLoss用に、バッチ次元と系列次元をまとめる。
    logits = logits.reshape(-1, vocab_size)

    # 予測と正解Token IDからLossを計算する。
    # targetに対応するTokenの確率に-log()をした値
    loss = criterion(logits, targets)

    # 前回のiterationで計算された勾配を消去する。
    optimizer.zero_grad()

    # 誤差逆伝播によって各パラメータの勾配を計算する。
    loss.backward()

    # 勾配を使ってパラメータを更新する。
    optimizer.step()

    # 損失を出力
    if epoch % 10 == 0:
        loss_train = loss.item()
        print(f"epoch = {epoch+1}, loss_train = {loss_train}")

# 学習済みモデルの保存
model_path = Path(__file__).with_name("small_transformer.pth")
torch.save(net.state_dict(), model_path)

print(f"モデルを保存しました: {model_path}")
