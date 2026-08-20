'''
" "と入力したら、
"吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
を出力するモデルになったかをテストする
'''

from pathlib import Path

import torch
from transformer import EncoderDecoder
from transformer import SmallTransformer

# GPUが必要
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train.pyと完全に同じ文章を使用する
y = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"

# train.pyと同じ文字とToken IDの対応を再構築する
encoder_decoder = EncoderDecoder(y)
vocab_size = encoder_decoder.get_vocab_size()

# train.pyと同じ構成のモデルを作る
net = SmallTransformer(vocab_size).to(device)

# 学習済みパラメータを読み込む
model_path = Path(__file__).with_name("small_transformer.pth")
state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True,
)
net.load_state_dict(state_dict)

# 推論モードにする
net.eval()

# 空白は学習時に使用した開始トークン
# 「 」まで入力済みとして、残りを生成する
tokens = encoder_decoder.encode(" ").unsqueeze(0).to(device)

with torch.inference_mode():
    # 「 」は入力済みなので、残り33文字を生成
    # len(y)をlen(y)*2とかにすると意味不明な日本語を生成できる
    for _ in range(len(y)):
        logits = net(tokens)

        # 最後の位置における、次の文字の予測を取得
        next_token_logits = logits[:, -1, :]

        # 各語彙のlogitを確率に変換する。
        # shape: (B, vocab_size) -> (B, vocab_size)
        probs = torch.softmax(next_token_logits, dim=-1)

        # 確率分布 probs に従って、各バッチから1つのToken IDをサンプリングする。
        # probs.shape = (B, vocab_size)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # 既存のToken列の末尾に、生成した1Tokenを追加する。
        # (B, T) + (B, 1) -> (B, T+1)
        tokens = torch.cat([tokens, next_token_id], dim=1)

        # 今回生成した文字を表示
        new_token = encoder_decoder.decode(tokens[0].cpu())
        print(new_token)

# バッチ次元を除き、CPUへ移動して文字列に戻す
generated_text = encoder_decoder.decode(tokens[0].cpu())

# 先頭の開始トークン（空白）を除外
generated_text = generated_text[1:]

print(f"生成結果: {generated_text}")
print(f"正解:     {y}")
print(f"一致:     {generated_text == y}")