#単語埋め込みの平均ベクトルでテキストの特徴ベクトルを表現し、重みベクトルとの内積でポジティブ及びネガティブを分類するニューラルネットワーク（ロジスティック回帰モデル）を設計せよ。

import torch
import torch.nn as nn

class NeuralNetworkModel(nn.Module):
    
    def __init__(self, embeddings): #モデルの部品
        
        super().__init__() #親クラスの初期化を呼び出す(ここではnn.Module)
        
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embeddings, dtype = torch.float32), #事前学習済み単語ベクトル行列を埋め込みとして用いる
            freeze = True, #学習の過程で単語ベクトルを更新しない
            padding_idx = 0 #パディング用インデックスは0
        )
        
        self.linear = nn.Linear(embeddings.shape[1], 1) #300次元の埋め込み行列から1次元の出力に変換する線形層
    
    def forward(self, input_ids): #入力から出力までの過程
        
        embeds = self.embedding(input_ids) #単語IDを埋め込みに変換
        text_vector = embeds.mean(dim = 1) #文中の単語ベクトルの平均を取る(単語の順序の情報が消滅)
        logit = self.linear(text_vector) #平均ベクトルを線形変換して1次元の分類スコアを計算(まだ確率ではない)
        
        return logit