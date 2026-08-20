import torch.nn as nn
import torch
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        # 親クラス nn.Module の初期化を行う。
        super().__init__()

        # モデル全体の特徴次元数を保存する。
        self.d_model = d_model

        # Attention headの数を保存する。
        self.num_heads = num_heads

        # 各headへ均等に分割できることを確認する。
        assert d_model % num_heads == 0

        # 1つのheadが担当する特徴次元数を計算する。
        self.d_head = d_model // num_heads

        # 入力XからQueryベクトルQを生成する線形変換。
        self.query = nn.Linear(d_model, d_model)

        # 入力XからKeyベクトルKを生成する線形変換。
        self.key = nn.Linear(d_model, d_model)

        # 入力XからValueベクトルVを生成する線形変換。
        self.value = nn.Linear(d_model, d_model)

        # 4つのheadから得られた特徴を統合するための出力線形層。
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 入力テンソル x の shape は (B, T, d_model)
        # バッチサイズとトークンの個数を取得
        B, T, _ = x.shape

        # 入力 x から Query を生成する。
        # shape: (B, T, d_model)
        Q = self.query(x)

        # 入力 x から Key を生成する。
        # shape: (B, T, d_model)
        K = self.key(x)

        # 入力 x から Value を生成する。
        # shape: (B, T, d_model)
        V = self.value(x)

        '''
        以下、Multi Head Attention
        '''

        # 64次元を4個のheadに分割する。
        # (B, T, 64) -> (B, T, self.num_heads, 16)
        Q = Q.reshape(B, T, self.num_heads, self.d_head)
        K = K.reshape(B, T, self.num_heads, self.d_head)
        V = V.reshape(B, T, self.num_heads, self.d_head)

        # head次元を前に移動する。
        # (B, T, self.num_heads, 16) -> (B, self.num_heads, T, 16)
        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        # 各Queryと各Keyの内積を計算し、
        # 値が大きくなりすぎないよう sqrt(d_head) でスケーリングする。
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal Maskを適用してからsoftmaxをかける
        # 下三角のCausal Maskを作る。
        # Trueの位置だけ参照を許可する。
        mask = torch.ones(T, T, device=scores.device, dtype=torch.bool).tril()

        # 未来のトークンに対応するスコアを -∞ にする。
        # scores.shape = (B, num_heads, T, T) だが、
        # mask.shape = (T, T) は broadcast される。
        scores = scores.masked_fill(~mask, float("-inf"))

        # 各Queryについて、参照可能なKey方向にsoftmaxを取る。
        A = nn.functional.softmax(scores, dim=-1)

        # Attention重み A を使って Value を重み付き平均する。
        # 各トークンxの新たな特徴ベクトルyを並べたもの
        # (B, 4, T, T) @ (B, 4, T, 16) -> (B, 4, T, 16)
        Y = A @ V

        # head次元とtoken次元を入れ替える。
        # (B, 4, T, 16) -> (B, T, 4, 16)
        # その後、4 head × 16次元を結合して64次元に戻す。
        Y = Y.transpose(-3, -2).reshape(B, T, self.d_model)

        # 各headを結合した特徴を線形変換し、head間の情報を混ぜ合わせる。
        Y = self.Wo(Y)
        return Y


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        # 親クラス nn.Module を初期化する。
        super().__init__()

        # Attention の前に適用する LayerNorm。
        self.norm1 = nn.LayerNorm(d_model)

        # FFN の前に適用する LayerNorm。
        self.norm2 = nn.LayerNorm(d_model)

        # 自作した Multi-Head Causal Self-Attention。
        self.attention = MultiHeadAttention(d_model, num_heads)

        # 各トークンを独立に変換する Feed Forward Network。
        # d_model -> 4*d_model -> d_model と変換する。
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # Attention の前に LayerNorm を適用する。
        x_normal = self.norm1(x)

        # Multi-Head Attention の出力を元の入力 x に残差接続する。
        y = x + self.attention(x_normal)

        # FFN の前に2つ目の LayerNorm を適用する。
        y_normal = self.norm2(y)

        # FFN の出力を y に残差接続する。
        z = y + self.ffn(y_normal)

        # Transformer Block の最終出力を返す。
        return z


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=64, num_heads=4, num_layers=4, max_len=128):
        # nn.Module の初期化を行う。
        super().__init__()

        self.max_len = max_len

        # 各Token IDをd_model次元のベクトルへ変換する。
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 0〜max_len-1の各位置について、
        # 学習可能なd_model次元の位置ベクトルを持つ。
        self.position_embedding = nn.Embedding(max_len, d_model)

        # num_layers個のTransformer Blockを生成する。
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 各トークン位置の d_model 次元表現を、
        # 語彙全体の logits に変換する出力層。
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        # tokens.shape = (B, T)
        B, T = tokens.shape

        # 入力系列長が最大系列長を超えていないことを確認する。
        assert T <= self.max_len

        # Token ID を埋め込みベクトルへ変換する。
        # (B, T) -> (B, T, d_model)
        token_vectors = self.token_embedding(tokens)

        # 位置ID [0, 1, ..., T-1] を生成する。
        # shape: (T,)
        position_id = torch.arange(T, device=tokens.device)

        # 位置IDを位置埋め込みベクトルへ変換する。
        # (T,) -> (T, d_model)
        position_vectors = self.position_embedding(position_id)

        # Token情報と位置情報を加算する。
        # (B, T, d_model) + (T, d_model)
        # -> (B, T, d_model)
        x = token_vectors + position_vectors

        # Transformer Blockを順番に通す。
        # 各Blockでshapeは (B, T, d_model) のまま。
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 各トークン位置から、語彙全体に対する次のトークンの予測logitsを計算する。
        # (B, T, d_model) -> (B, T, vocab_size)
        logits = self.output(x)

        # 次トークン予測用のlogitsを返す。
        return logits

class EncoderDecoder:
    def __init__(self, text: str):
        # 入力テキストを保存
        self.text = text

        # Vocabularyを事前に決め打ち。
        text += ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?! \
        あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもらりるれろやゆよわをん。、'

        # 重複がない配列にする
        chars = sorted(set(text))

        # 文字 -> Token ID の対応表
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # Token ID -> 文字 の対応表
        self.itos = {i: ch for i, ch in enumerate(chars)}

    # 文字列 -> Token IDのTensor
    def encode(self, text) -> torch.Tensor:
        # 入力された文字列を整数の配列にして返す
        token_ids = [self.stoi[ch] for ch in text]
        return torch.tensor(token_ids, dtype=torch.long)

    # Token IDのTensor -> 文字列
    def decode(self, token_ids: torch.Tensor):
        # まずTensorからリストに変換する
        token_ids = token_ids.tolist()

        txt = [self.itos[num] for num in token_ids]
        outputs = "".join(txt)
        return outputs

    def get_vocab_size(self):
        return len(self.stoi)
