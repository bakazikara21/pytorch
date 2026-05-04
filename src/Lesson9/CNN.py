# 必要ライブラリのインポート
# %%
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
# torch関連ライブラリのインポート

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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

# データの準備
# transformでテンソル化 + データ正規化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
])

# datasetからCIFAR-10の画像を取得
# トロント大学のurl = https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
data_root = './data'
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

image, label = train_set[0]

# データローダーの定義
batch_size = 100        # 100枚の画像を1グループとして学習を行う

# 訓練用データローダー
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)

# テスト用データローダー
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)

# 正解ラベルの定義->リストとして定義
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# %%
# 次に使いまわすための共通関数を定義する
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# fit()関数 := 学習を行う関数でhistoryを返す。追加学習がしやすいような設計になっている
def fit(net,criterion,optimizer,epochs,history,device):
    # tqdmライブラリのインポート
    from tqdm.notebook import tqdm
    base_epoch = len(history)
    for epoch in range(base_epoch,epochs+base_epoch):
        # epochs回、学習を回す
        # 1エポックあたりの正解数
        n_train_acc , n_test_acc = 0,0

        # 1エポックあたりの合計損失
        n_train_loss, n_test_loss = 0.0,0.0

        # 1エポックあたりの合計データ数
        n_train, n_test = 0,0

        # 訓練フェーズを明示する
        net.train()
        for images,labels in tqdm(train_loader):
            # 1バッチあたりのデータ件数
            train_batch_size = len(labels)
            n_train += train_batch_size

            # 勾配の初期化
            optimizer.zero_grad()

            # 入力データとラベルをGPUに送ってから出力を売る
            inputs = images.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            # 損失の計算
            loss = criterion(outputs,labels)

            # 勾配の計算
            loss.backward()

            # パラメータの更新
            optimizer.step()

            # 予測結果
            predict = torch.max(outputs,1)[1] # 予測ラベルのベクトルになる

            # 正解数の加算
            n_train_acc += (predict == labels).sum().item()

            # 損失の加算
            n_train_loss += loss.item() * train_batch_size

        # 訓練フェーズを明示する
        net.eval()
        for images_test,labels_test in tqdm(test_loader):
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels)
            n_test += test_batch_size

            # 入力データとラベルをGPUに送ってから出力を売る
            inputs = images_test.to(device)
            labels = labels_test.to(device)
            outputs = net(inputs)

            # 損失の計算
            loss = criterion(outputs,labels)

            # 予測結果
            predict = torch.max(outputs,1)[1] # 予測ラベルのベクトルになる

            # 正解数の加算
            n_test_acc += (predict == labels).sum().item()

            # 損失の加算
            n_test_loss += loss.item() * test_batch_size

        # 1エポックが終了したので、損失と精度を計算する
        train_loss = n_train_loss / n_train
        test_loss  = n_test_loss  / n_test

        train_acc = n_train_acc / n_train
        test_acc = n_test_acc   / n_test 

        item = np.array([epoch+1,train_loss,train_acc,test_loss,test_acc])
        history = np.vstack((history,item))
        print(f'エポック数：{epoch+1}, テストデータの損失：{test_loss:.5f}, テストデータの精度：{test_acc:.5f}')
    return history

# 学習ログ解析->学習後にネットワークの性能を可視化する関数
def evaluate_history(history,png_loss, png_acc):
    #損失と精度の確認
    print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}') 
    print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='訓練')
    plt.plot(history[:,0], history[:,3], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('損失')
    plt.title('学習曲線(損失)')
    plt.legend()
    plt.savefig('./data/picture/' + png_loss)
    plt.show()

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='訓練')
    plt.plot(history[:,0], history[:,4], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('精度')
    plt.title('学習曲線(精度)')
    plt.legend()
    plt.savefig('./data/picture/' + png_acc)
    plt.show()

# イメージとラベル表示
def show_images_labels(loader, classes, net, device):
    # データローダーから最初の1セットを取得する
    for images, labels in loader:
        break
    # 表示数は50個とバッチサイズのうち小さい方
    n_size = min(len(images), 50)

    if net is not None:
      # デバイスの割り当て
      inputs = images.to(device)
      labels = labels.to(device)

      # 予測計算
      outputs = net(inputs)
      predicted = torch.max(outputs,1)[1]
      #images = images.to('cpu')

    # 最初のn_size個の表示
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # netがNoneでない場合は、予測結果もタイトルに表示する
        if net is not None:
          predicted_name = classes[predicted[i]]
          # 正解かどうかで色分けをする
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # netがNoneの場合は、正解ラベルのみ表示
        else:
          ax.set_title(label_name, fontsize=20)
        # TensorをNumPyに変換
        image_np = images[i].numpy().copy()
        # 軸の順番変更 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 値の範囲を[-1, 1] -> [0, 1]に戻す
        img = (img + 1)/2
        # 結果表示
        plt.imshow(img)
        ax.set_axis_off()
    plt.savefig('./data/picture/result_visualize.png')
    plt.show()


# ここからはモデル定義
# CNNのモデル定義 全結合層を含むことに注意
class CNN(nn.Module):
    def __init__(self, n_output,n_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3)  # (入力チャネル,出力チャネル,カーネルの一辺の長さ)
        self.conv2 = nn.Conv2d(32,32,3) # 2つ目の畳み込み層
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))  # 2*2の領域の最大値をスライディングして取得
        self.flatten = nn.Flatten()         # 1次元ベクトル化
        self.l1 = nn.Linear(6272,n_hidden)  # 1次元ベクトルを全結合層に入力
        self.l2 = nn.Linear(n_hidden,n_output)  # ソフトマックスにかける前の出力

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool
        )
        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )
    
    def forward(self,x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3

# 学習に必要な変数の初期化
n_output = len(classes)
n_hidden = 128  # 全結合層の隠れ層のノード数

# 乱数初期化
torch_seed()
net = CNN(n_output,n_hidden).to(device)

epochs = 50
eps = 0.01
optimizer = optim.SGD(net.parameters(),lr=eps)

criterion = nn.CrossEntropyLoss()

history = np.zeros((0,5))

# %% 
# 以下、fit()関数を使って学習を行う
history = fit(net,criterion,optimizer,epochs,history,device)

# %%
# 学習結果を表示
evaluate_history(history,'result_loss.png','result_acc.png')

# %%
# 最初の50個の表示
show_images_labels(test_loader, classes, net, device)

# %%
