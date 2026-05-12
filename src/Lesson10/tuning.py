# ニューラルネットワークのチューニング
# 1.ニューラルネットワークの多層化
# 2.最適化アルゴリズムの改善（モメンタムやAdam）
# 3.過学習の対策（バッチ正規化やドロップアウト）

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
transform = transforms.Compose([
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
    transform=transform
)
test_set = datasets.CIFAR10(
    root = data_root,
    train=False,
    download=True, # トロント大学のサーバーがダウンしているとダウンロードできないことに注意
    transform=transform
)

# image, label = train_set[0] # 1枚目の画像とラベル

# データローダーの定義
batch_size = 100        # 100枚の画像を1グループとして学習を行う

# 訓練用データローダー　5万枚の画像データをバッチサイズで分割する
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)

# テスト用データローダー  1万枚の画像データをバッチサイズで分割する
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)

# 正解ラベルの定義->リストとして定義
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# %%
# 以下でGitHubからダウンロードした共通関数ライブラリを使用する宣言
# これにより、fit()やevaluate_history()などが使える
# git clone https://github.com/makaishi2/pythonlibs.git
from pythonlibs.torch_lib1 import *

# %%
# クラス定義 -> ニューラルネットワークの多層化をして精度の変化を観察する
class CNN_v2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=(1,1)) # 入力チャネルは3 出力チャネルは32とする
        self.conv2 = nn.Conv2d(32,32,3,padding=(1,1)) # 2つ目の畳み込み層
        self.conv3 = nn.Conv2d(32,64,3,padding=(1,1)) # 3つ目の畳み込み層
        self.conv4 = nn.Conv2d(64,64,3,padding=(1,1)) # 4つ目の畳み込み層
        self.conv5 = nn.Conv2d(64,128,3,padding=(1,1)) # 5つ目の畳み込み層
        self.conv6 = nn.Conv2d(128,128,3,padding=(1,1)) # 6つ目の畳み込み層

        self.maxpool = nn.MaxPool2d((2,2))      # 最大プーリングにより縦横のサイズが1/2になる
        self.flatten = nn.Flatten()             # 畳み込み層から全結合層へ入力するために1次元ベクトル化
        self.l1 = nn.Linear(4*4*128,128)        # 4*4のデータが128チャネルあるのでこうなります。
        self.l2 = nn.Linear(128,num_classes)    # 多クラス分類のクラス数と同じだけ出力する
        self.relu = nn.ReLU(inplace=True)       # 値を保存しない設定にする

        self.features = nn.Sequential(
            # この5つがひとかたまりみたいになってる
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool,

            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.maxpool,

            self.conv5,
            self.relu,
            self.conv6,
            self.relu,
            self.maxpool,
        )
        self.classifies = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )

    def forward(self,x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifies(x2)
        return x3
    
# 以下は様々な変数の初期化
torch_seed()
num_classes = len(list(set(classes))) # 分類先クラス数　今回は10になる
net = CNN_v2(num_classes).to(device)
epochs = 50     # エポック数
eps    = 0.01   # 学習率
optimizer = optim.SGD(net.parameters(),lr=eps)
criterion = nn.CrossEntropyLoss()
history = np.zeros((0,5))

# %%
# 以下、学習開始
history = fit(net,optimizer,criterion,epochs,train_loader,test_loader,device,history)

# %%
# 学習曲線の可視化
evaluate_history(history)

# %%
# 次に最適化アルゴリズムの改善を行う
# モメンタムSGDとAdamを比較する
torch_seed()
net = CNN_v2(num_classes).to(device)
lr = 0.01
epochs = 20 # はやく収束するので20回で十分
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)    # モメンタムSGD
history2 = np.zeros((0,5))

# %%
# モメンタムSGDの学習開始
history2 = fit(net,optimizer,criterion,epochs,train_loader,test_loader,device,history2)

# %%
# モメンタムSGDの学習曲線の可視化
evaluate_history(history2)

# %%
# 次にAdamを評価する
torch_seed()
epochs = 20 # はやく収束するので20回で十分
net = CNN_v2(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())    # モメンタムSGD
history3 = np.zeros((0,5))

# %%
# Adamでの学習開始
history3 = fit(net,optimizer,criterion,epochs,train_loader,test_loader,device,history3)

# %% 
# Adamの学習曲線の可視化
evaluate_history(history3)

# %%
# ドロップアウトを行い、過学習を抑制・精度向上を目指す
# CNNにドロップアウトを追加すればよい
class CNN_v3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=(1,1)) # 入力チャネルは3 出力チャネルは32とする
        self.conv2 = nn.Conv2d(32,32,3,padding=(1,1)) # 2つ目の畳み込み層
        self.conv3 = nn.Conv2d(32,64,3,padding=(1,1)) # 3つ目の畳み込み層
        self.conv4 = nn.Conv2d(64,64,3,padding=(1,1)) # 4つ目の畳み込み層
        self.conv5 = nn.Conv2d(64,128,3,padding=(1,1)) # 5つ目の畳み込み層
        self.conv6 = nn.Conv2d(128,128,3,padding=(1,1)) # 6つ目の畳み込み層

        self.maxpool = nn.MaxPool2d((2,2))      # 最大プーリングにより縦横のサイズが1/2になる
        self.flatten = nn.Flatten()             # 畳み込み層から全結合層へ入力するために1次元ベクトル化
        self.l1 = nn.Linear(4*4*128,128)        # 4*4のデータが128チャネルあるのでこうなります。
        self.l2 = nn.Linear(128,num_classes)    # 多クラス分類のクラス数と同じだけ出力する
        self.relu = nn.ReLU(inplace=True)       # 値を保存しない設定にする

        self.dropout1 = nn.Dropout(0.2) # 20%のノードの出力を0にする
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)

        self.features = nn.Sequential(
            # この5つがひとかたまりみたいになってる
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool,
            self.dropout1,

            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.maxpool,
            self.dropout2,

            self.conv5,
            self.relu,
            self.conv6,
            self.relu,
            self.maxpool,
            self.dropout3
        )
        self.classifies = nn.Sequential(
            self.l1,
            self.relu,
            self.dropout3,
            self.l2
        )

    def forward(self,x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifies(x2)
        return x3
    
# %%
# 次にDropoutをしてみる
torch_seed()
epochs = 50
net = CNN_v3(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())    # モメンタムSGD
history4 = np.zeros((0,5))

# %%
# dropoutありで学習開始
history4 = fit(net,optimizer,criterion,epochs,train_loader,test_loader,device,history4)

# %%
evaluate_history(history4)

# %%
# 続いて、Batch Normalization
# 畳み込み層の出力を活性化させる前にバッチ正規化
# したがって、CNNにバッチ正規化を追加すればよい
# バッチ正規化は、正規化した各出力をweight倍してbiasを足すので
# パラメータを持っており、学習の対象であることに注意
class CNN_v4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=(1,1)) # 入力チャネルは3 出力チャネルは32とする
        self.conv2 = nn.Conv2d(32,32,3,padding=(1,1)) # 2つ目の畳み込み層
        self.conv3 = nn.Conv2d(32,64,3,padding=(1,1)) # 3つ目の畳み込み層
        self.conv4 = nn.Conv2d(64,64,3,padding=(1,1)) # 4つ目の畳み込み層
        self.conv5 = nn.Conv2d(64,128,3,padding=(1,1)) # 5つ目の畳み込み層
        self.conv6 = nn.Conv2d(128,128,3,padding=(1,1)) # 6つ目の畳み込み層

        self.maxpool = nn.MaxPool2d((2,2))      # 最大プーリングにより縦横のサイズが1/2になる
        self.flatten = nn.Flatten()             # 畳み込み層から全結合層へ入力するために1次元ベクトル化
        self.l1 = nn.Linear(4*4*128,128)        # 4*4のデータが128チャネルあるのでこうなります。
        self.l2 = nn.Linear(128,num_classes)    # 多クラス分類のクラス数と同じだけ出力する
        self.relu = nn.ReLU(inplace=True)       # 値を保存しない設定にする

        self.dropout1 = nn.Dropout(0.2) # 20%のノードの出力を0にする
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)

        self.bn1 = nn.BatchNorm2d(32)   # 1つ目のバッチ正規化
        self.bn2 = nn.BatchNorm2d(32)   # 2つ目のバッチ正規化
        self.bn3 = nn.BatchNorm2d(64)   # 3つ目のバッチ正規化
        self.bn4 = nn.BatchNorm2d(64)   # 4つ目のバッチ正規化
        self.bn5 = nn.BatchNorm2d(128)  # 5つ目のバッチ正規化
        self.bn6 = nn.BatchNorm2d(128)  # 6つ目のバッチ正規化

        self.features = nn.Sequential(
            # この5つがひとかたまりみたいになってる
            self.conv1, self.bn1,
            self.relu,
            self.conv2, self.bn2,
            self.relu,
            self.maxpool,
            self.dropout1,

            self.conv3, self.bn3,
            self.relu,
            self.conv4, self.bn4,
            self.relu,
            self.maxpool,
            self.dropout2,

            self.conv5, self.bn5,
            self.relu,
            self.conv6, self.bn6,
            self.relu,
            self.maxpool,
            self.dropout3
        )
        self.classifies = nn.Sequential(
            self.l1,
            self.relu,
            self.dropout3,
            self.l2
        )

    def forward(self,x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifies(x2)
        return x3
    
# %%
# 次にBatch Normalizationをしてみる
torch_seed()
epochs = 50
net = CNN_v4(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
history5 = np.zeros((0,5))

# %%
# バッチ正規化を組み込んで学習開始
history5 = fit(net,optimizer,criterion,epochs,train_loader,test_loader,device,history5)

# %%
# バッチ正規化を適用した結果を可視化
evaluate_history(history5)
# %%
# 最後にData Augmentationを施す。
# これは、クラス分類が変わらない範囲でのデータの加工をし、
# さらに頑健なニューラルネットワークを作れる
# 学習回数が増えることに注意
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # 左右反転の加工を施す
    transforms.ToTensor(),  # テンソル化
    transforms.Normalize(0.5,0.5), # [-1,1]に正規化
    transforms.RandomErasing(p=0.5, scale=(0.02,0.33),
                             ratio=(0.3,3.3), value=0, inplace=False), # 画像の遮蔽
])
train_set2 = datasets.CIFAR10(
    root=data_root, train=True, 
    download=True, transform= transform_train
)
batch_size = 100
train_loader2 = DataLoader(train_set2, batch_size=batch_size,shuffle=True)

torch_seed()
show_images_labels(train_loader2,classes,None,None) # 加工後のデータがわかるはず

torch_seed()
net = CNN_v4(num_classes).to(device)
epochs = 100    # 学習に時間がかかる
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
history6 = np.zeros((0,5))
# %%
# データ加工を施す学習開始
history6 = fit(net,optimizer,criterion,epochs,train_loader2,test_loader,device,history6)

# %%
evaluate_history(history6)
# %%
# 実際のテストデータに対する挙動を見てみる
show_images_labels(test_loader,classes,net,device)
# %%
