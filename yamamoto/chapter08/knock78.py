#問題77の学習において、単語埋め込みのパラメータも同時に更新するファインチューニングを導入せよ。また、学習したモデルの開発セットにおける正解率を求めよ。

#実行はGoogleColab(GPU:T4)

#from google.colab import drive
from gensim.models import KeyedVectors
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim

#drive.mount("/content/drive")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#単語埋め込み行列の作成

model = KeyedVectors.load_word2vec_format(
    "/content/drive/MyDrive/nlp100knock/GoogleNews-vectors-negative300.bin.gz",
    binary = True,
    limit = 1500000 #メモリに応じて単語数を制限
)

v = len(model.index_to_key) + 1 #埋め込み行列の行数(語彙+1)
d = model.vector_size #埋め込み行列の列数

embeddings = np.zeros((v, d), dtype = np.float32) #サイズが(v,d)のゼロ行列

word_to_id = {"<PAD>": 0} #wordからidを対応づける辞書
id_to_word = {0: "<PAD>"} #idからwordを対応づける辞書

for i, word in enumerate(model.index_to_key, start = 1): #i=1スタートで単語とそのインデックスを取り出す

    embeddings[i] = model[word] #wordの埋め込みを埋め込み行列のi行目へ
    word_to_id[word] = i #wordにインデックスiを対応
    id_to_word[i] = word #インデックスiにwordを対応

print("embedding_shape:", embeddings.shape)
print("United States:", word_to_id["United_States"])
print("ID_410:", id_to_word[410])

#データセットの準備

train_data = []
dev_data = []

with open("/content/drive/MyDrive/nlp100knock/SST-2/train.tsv", "r", encoding = "utf-8") as file: #訓練データ用

    reader = csv.reader(file, delimiter = "\t")
    next(reader)

    for line in reader:

        text_dict = {} #データの1つのテキストについての情報を格納する辞書

        text_dict["text"] = line[0] #textはtabで分割した1つめ
        text_dict["label"] = torch.tensor([float(line[1])]) #ラベルはtabで分割した2つめ

        text = line[0].split() #textを分割

        input_ids = [] #分割された各単語のIDを格納

        for token in text:

            if token in word_to_id: #その単語のIDが存在する場合

                id_number = word_to_id[token]
                input_ids.append(id_number)

        if len(input_ids) == 0: #全ての単語にIDが存在しなければ飛ばす

            continue

        text_dict["input_ids"] = torch.tensor(input_ids)
        train_data.append(text_dict)

with open("/content/drive/MyDrive/nlp100knock/SST-2/dev.tsv", "r", encoding = "utf-8") as file: #開発データ用

    reader = csv.reader(file, delimiter = "\t")
    next(reader)

    for line in reader:

        text_dict = {}

        text_dict["text"] = line[0]
        text_dict["label"] = torch.tensor([float(line[1])])

        text = line[0].split()

        input_ids = []

        for token in text:

            if token in word_to_id:

                id_number = word_to_id[token]
                input_ids.append(id_number)

        if len(input_ids) == 0:

            continue

        text_dict["input_ids"] = torch.tensor(input_ids, dtype = torch.long)
        dev_data.append(text_dict)

print(train_data[:2])
print(dev_data[:2])

#ニューラルネットワークモデルの設計

class NeuralNetworkModel(nn.Module):

    def __init__(self, embedding_matrix): #モデルの部品

        super().__init__() #親クラスの初期化を呼び出す(ここではnn.Module)

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype = torch.float32), #事前学習済み単語ベクトル行列を埋め込みとして用いる
            freeze = False, #学習の過程で単語ベクトルを更新(ファインチューニング)
            padding_idx = 0 #パディング用インデックスは1
        )

        self.linear = nn.Linear(embedding_matrix.shape[1], 1) #300次元の埋め込み行列から1次元の出力に変換する線形層

    def forward(self, input_ids): #入力から出力までの過程

        embeds = self.embedding(input_ids) #単語IDを埋め込みに変換
        text_vector = embeds.mean(dim = 1) #文中の単語ベクトルの平均を取る
        logit = self.linear(text_vector) #平均ベクトルから1次元の分類スコアを計算(まだ確率ではない)

        return logit
    
#複数の事例をまとめてパディングする関数collate

def collate(data):

    padded = {} #パディングしたデータの辞書

    input_ids = []
    label = []

    sorted_data = sorted(data, key = lambda x: len(x["input_ids"]), reverse = True) #dataをIDトークン列の長い順にソート

    max_len = len(sorted_data[0]["input_ids"]) #最長のトークン列の長さを保存

    for example in sorted_data:

        pad = torch.tensor([0] * (max_len - len(example["input_ids"])), dtype = torch.long) #最長の長さに合わせてパディング要素(0)を作成(torch.long:整数テンソル型)

        input_ids.append(torch.cat([example["input_ids"], pad])) #パディング要素を追加(torch.cat:複数のテンソルを結合)
        label.append(example["label"])

    padded["input_ids"] = torch.stack(input_ids) #tensorのリストを結合して1つのtensorにまとめる
    padded["label"] = torch.stack(label)

    return padded

print(collate(train_data[:5]))

#GPUを用いてミニバッチ学習

train_batch = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = 128,
    shuffle = True,
    collate_fn = collate
)

model = NeuralNetworkModel(embeddings)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(10):

    total_loss = 0.0

    for batch in train_batch:

        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_batch)
    print(epoch, "損失平均：", avg_loss)

model.eval()

score = 0

#開発データで評価

with torch.no_grad():

    for example in dev_data:

        input_ids = example["input_ids"].unsqueeze(0).to(device)
        label = float(example["label"])

        logit = model(input_ids) #モデルで分類スコアを計算

        prob = torch.sigmoid(logit) #シグモイド関数で確率に変換

        pred = (prob >= 0.5).float() #0.5以上で1.0(True), それ以下は0.0(False)でラベルを予測

        if pred.item() == label:

            score += 1

    accuracy = score / len(dev_data)

print("正解率：", accuracy)