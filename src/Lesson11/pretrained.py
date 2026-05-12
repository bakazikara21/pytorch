# ニューラルネットワークの事前学習済みのモデル
# 1.転移学習
# 2.ファインチューニング <- こっちのほうが精度がよかったからこっちをつかう
# ファインチューニング：
# 出力層を除く学習済みのモデルのパラメータ（全体 or 一部）を初期値として、後半の全結合層を追加して
# （全体 or 一部）のパラメータを再学習する。追加した後半の層のパラメータは必ず再学習することに注意。
# 転移学習：
# 学習済みのモデルを特徴抽出器として利用し、
# それを使って得られた特徴を入力として、求める出力が得られる予測器のパラメータのみ追加で学習させればよい。
# ファインチューニングの一部とも捉えられる。ただし、予測器はNNではなく、サポートベクターマシンや決定木などでよい

# 必要ライブラリのインポート
# %%
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import display
# torch関連ライブラリのインポート

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

# 以下でGitHubからダウンロードした共通関数ライブラリを使用する宣言
# これにより、fit()やevaluate_history()などが使える
# git clone https://github.com/makaishi2/pythonlibs.git
import os
import sys

# 現在のディレクトリの親ディレクトリ（src）をパスに追加
sys.path.append(os.path.abspath(".."))
from pythonlibs.torch_lib1 import *

# warning表示off
import warnings
warnings.simplefilter('ignore')

# %%

# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# numpyの表示桁数設定
np.set_printoptions(suppress=True, precision=5)

# plt.savefig("./data/picture/sin.png")で画像を保存できる
# デバイスの割り当て
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device) # cuda:0と出力されるはず

# %%
# データの準備
# transformでテンソル化 + データ正規化
transform_train = transforms.Compose([
    transforms.Resize(112), # 比率を維持したまま、短いほうの辺を112画素とする
    transforms.RandomHorizontalFlip(p=0.5), # 左右反転の加工を施す
    transforms.ToTensor(),  # テンソル化
    transforms.Normalize(0.5,0.5), # [-1,1]に正規化
    transforms.RandomErasing(p=0.5, scale=(0.02,0.33),
                             ratio=(0.3,3.3), value=0, inplace=False), # 画像の遮蔽
])
transform = transforms.Compose([
    transforms.Resize(112), # 224 * 448 ならば 112 * 224の画像に変換される
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
])

# datasetからCIFAR-10の画像を取得
# トロント大学のurl = https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
data_root = '../data'
train_set = datasets.CIFAR10(
    root = data_root,
    train=True,
    download=True, # トロント大学のサーバーがダウンしているとダウンロードできないことに注意
    transform=transform_train
)
test_set = datasets.CIFAR10(
    root = data_root,
    train=False,
    download=True, # トロント大学のサーバーがダウンしているとダウンロードできないことに注意
    transform=transform
)

# image, label = train_set[0] # 1枚目の画像とラベル

# データローダーの定義
batch_size = 50        # 50枚の画像を1グループとして学習を行う->事前学習済みのモデルが巨大なので小さくした

# 訓練用データローダー　5万枚の画像データをバッチサイズで分割する
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)

# テスト用データローダー  1万枚の画像データをバッチサイズで分割する
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)

# 正解ラベルの定義->リストとして定義
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# %%
# 事前学習済みのモデルResNet-18を読み込む(2行)
from torchvision import models
net = models.resnet18(pretrained = True)
net = net.to(device)
summary(net,(100,3,112,112)) # AdaptiveAvgPool -> GAP層:各チャネルの平均をとってきて、チャネル数の要素を持つ1次元ベクトル化
# %%
print(net.fc) # 最後の全結合層 -> 入力が512、出力が1000の全結合層だとわかる

# %%
# net.fcを変更し、事前学習済みのモデルをファインチューニングしよう
torch_seed()
net = models.resnet18(pretrained = True)
# 最終レイヤの入力次元数を取得
fc_in_features = net.fc.in_features
n_output = len(classes)

# 最終レイヤを上書きしてファインチューニングの準備する
net.fc = nn.Linear(fc_in_features,n_output) # こいつはCPUにいるので、.to(device)が必須
net = net.to(device)

# 学習率
lr = 0.001
criterion = nn.CrossEntropyLoss() # 損失関数はsoftmax + 交差エントロピー関数
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9) # モメンタムSGDのような複雑ではないものを使うべき
history = np.zeros((0,5))
num_epochs = 5

# %%
# resNetのファインチューニングを開始 -> 精度94%達成！
history = fit(net,optimizer,criterion,num_epochs,train_loader,test_loader,device,history)
evaluate_history(history)

# %%
# 事前学習済みのモデルVGG-19-BNを読み込む(2行)
from torchvision import models
net = models.vgg19_bn(pretrained = True)
print(net.classifier[6]) # 最後の全結合層の形状：入力4096 出力1000の全結合層

# %%
# net.classifier[6]を変更し、事前学習済みのモデルをファインチューニングしよう
torch_seed()
net = models.vgg19_bn(pretrained = True)
# 最終レイヤの入力次元数を取得
fc_in_features = net.classifier[6].in_features
n_output = len(classes)

# 最終レイヤを上書きしてファインチューニングの準備する
net.classifier[6] = nn.Linear(fc_in_features,n_output) # こいつはCPUにいるので、.to(device)が必須
net.features = net.features[:-1] # 最大プーリングを外す->画像が小さいため、最大プーリングで小さくなりすぎてしまうので外す
net.avgpool = nn.Identity()      # 7*7の画像に変換するGAP層だが、画像が小さいため意味がないので単位行列にしておく
net = net.to(device)

# 学習率
lr = 0.001
criterion = nn.CrossEntropyLoss() # 損失関数はsoftmax + 交差エントロピー関数
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9) # モメンタムSGDのような複雑ではないものを使うべき
history2 = np.zeros((0,5))
num_epochs = 5

# %%
# VGG-19のファインチューニングを開始 -> 精度95.8%達成！
history2 = fit(net,optimizer,criterion,num_epochs,train_loader,test_loader,device,history2)
evaluate_history(history2)
# %%
