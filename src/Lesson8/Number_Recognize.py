# 必要ライブラリのインポート
# %%
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
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

# plt.savefig("data/sin.png")で画像を保存できる
# デバイスの割り当て
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device) # cuda:0と出力されるはず

# まず、データの準備
# 今回はMNISTの数字認識を行う
import torchvision.datasets as datasets

data_root = './data'    # データのルートディレクトリの定義

train_set0 = datasets.MNIST(
    # どこにデータをダウンロードするか
    root = data_root,

    # 訓練データとしてデータを受け取りますか？
    train=True,

    # データがダウンロードされていない場合、ダウンロードしますか?
    download=True # ダウンロードしたらFalseにしておく
)
# データの確認->OK
# image, label = train_set0[0]
# plt.title(f'{label}')
# plt.imshow(image,cmap='gray_r')
# plt.axis('off')
# plt.savefig('./data/picture/check5.png')

# Transformsによるデータの前処理
import torchvision.transforms as transforms

transform = transforms.Compose([
    # データのテンソル化
    transforms.ToTensor(),

    # データの正規化[0,1]->[-1,1]に正規化　引数は(平均,標準偏差)
    transforms.Normalize(0.5,0.5),

    # 入力テンソルの1階テンソル化->NNに入力しやすくなる
    transforms.Lambda(lambda x: x.view(-1)),
])

train_set = datasets.MNIST(
    root=data_root, train=True, download=True,
    transform=transform
)
test_set = datasets.MNIST(
    root=data_root, train=False, download=True,
    transform=transform
)

# ミニバッチ用データ生成
from torch.utils.data import DataLoader

# ミニバッチサイズを指定
batch_size = 500    # 500枚のデータを1グループとして学習する

# 訓練用データローダー 訓練用なのでシャッフル必要
train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False)

# %%
# こっからは隠れ層1つのニューラルネットワークを作るだけ
class Net(nn.Module):
    def __init__(self,n_input,n_output,n_hidden):
        super().__init__()

        # 中間層1つのニューラルネットワーク
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)

        self.relu = nn.ReLU(inplace=True) # 中間層の活性化関数

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3

image,labels = train_set[0]  # 1枚の入力データを抽出
n_input = image.shape[0]    # 入力データの次元
n_output = 10   # 出力データのクラス数
n_hidden = 128  # 中間層のノードの数を128に設定
# print(f'n_input:{n_input}, n_output:{n_output}')

# 乱数の固定化
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True
net = Net(n_input,n_output,n_hidden).to(device)    # ネットワークの作成
criterion = nn.CrossEntropyLoss()   # 多クラス分類なので(ソフトマックスを含んでいることに注意)

epochs = 100    # エポック数
eps = 0.01  # 学習率
optimizer = optim.SGD(net.parameters(), lr=eps) # 最適化アルゴリズム

history = np.zeros((0,5))  # (エポック数, train_loss, train_acc, test_loss, test_acc)

# %%
from tqdm.notebook import tqdm

for epoch in range(epochs):
    # 1エポックあたりの正解数
    n_train_acc, n_test_acc = 0,0
    
    # 1エポックあたりの累積損失->これを平均して損失とする
    train_loss, test_loss = 0, 0

    # 1エポックあたりのデータ累積件数
    n_train, n_test = 0, 0

    for images,labels in tqdm(train_loader):
        # 1バッチあたりのデータ件数
        train_batch_size = len(labels)

        # 1エポックあたりのデータ累積件数
        n_train += train_batch_size

        inputs = images.to(device)
        labels = labels.to(device)
        # 勾配の初期化
        optimizer.zero_grad()

        # 入力データの順伝播出力
        outputs = net(inputs)

        # 予測結果(整数ラベル)に変換
        predict = torch.max(outputs,1)[1]   # 最大値をとる添え字ベクトル

        # 損失の計算
        loss = criterion(outputs, labels)

        # 勾配の計算
        loss.backward()

        # パラメータの更新
        optimizer.step()

        # 訓練データの正解数を加算
        n_train_acc += (predict == labels).sum().item()

        # 訓練データの損失を加算
        train_loss += loss.item() * train_batch_size
    for images,labels in test_loader:
        ################################
        #
        #   以下はテストデータの損失と精度
        #
        ################################
        # 1バッチあたりのデータ件数
        test_batch_size = len(labels)

        # 1エポックあたりのデータ累積件数
        n_test += test_batch_size

        inputs = images.to(device)
        labels = labels.to(device)
        # 入力データの順伝播出力
        outputs = net(inputs)

        # 予測結果(整数ラベル)に変換
        predict = torch.max(outputs,1)[1]   # 最大値をとる添え字ベクトル

        # テストデータの正解数を加算
        n_test_acc += (predict == labels).sum().item()

        # テストデータの損失を加算
        test_loss += loss.item() * test_batch_size

    # 1エポックあたりの損失と精度をあらためて計算する
    ave_train_loss = train_loss / n_train
    ave_test_loss  = test_loss  / n_test

    train_acc = n_train_acc / n_train
    test_acc  = n_test_acc  / n_test
    # 結果表示
    print(f'epoch:{epoch+1}, テストデータの損失:{ave_test_loss:.5f}, テストデータの精度:{test_acc:.5f}')
    item = np.array([epoch+1,ave_train_loss,train_acc,ave_test_loss,test_acc])
    history = np.vstack((history,item))

# %%
# 学習した結果の可視化して終わり
plt.plot(history[:,0],history[:,1],c='k',label='訓練データの損失')
plt.plot(history[:,0],history[:,3],c='b',label='テストデータの損失')
plt.xlabel('エポック数')
plt.ylabel('損失')
plt.title('数字認識のNNの損失の推移')
plt.legend()
plt.savefig('./data/picture/result_loss.png')
plt.show()

plt.plot(history[:,0],history[:,2],c='k',label='訓練データの精度')
plt.plot(history[:,0],history[:,4],c='b',label='テストデータの精度')
plt.xlabel('エポック数')
plt.ylabel('精度')
plt.title('数字認識のNNの精度の推移')
plt.legend()
plt.savefig('./data/picture/result_accuracy.png')
plt.show()

# %%
import os
print(os.getcwd())

# %%
plt.figure(figsize=(10,3))
for images,labels in test_loader:
    # 500枚の画像と、そのラベル [images,labels]
    for i in range(20):
        ax = plt.subplot(2,10,i+1)  # 2行10列の枠に画像を貼っていく

        image = images[i]
        label = labels[i]
        ##########################################################
        #
        #   以下はテストデータの予測ラベルと実際のラベルを図示するコード
        #
        ##########################################################

        image = image.to(device)
        label = label.to(device)
        # 入力データの順伝播出力
        outputs = net(image)

        # 予測結果(整数ラベル)に変換
        predict = torch.max(outputs,0)[1].long()   # 最大値をとる添え字

        if (predict == label):
            c = 'b'
        else:
            c = 'r'
            
        # imgの範囲を[0, 1]に戻す
        image2 = (image + 1)/ 2
            
        # イメージ表示
        plt.imshow(image2.cpu().detach().reshape(28, 28),cmap='gray_r')
        ax.set_title(f'{label}:{predict}', c=c)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("./data/picture/result_visualize.png")
    plt.show()
    break
# %%
