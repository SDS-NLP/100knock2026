#問題72で設計したモデルの重みベクトルを訓練セット上で学習せよ。ただし、学習中は単語埋め込み行列の値を固定せよ（単語埋め込み行列のファインチューニングは行わない）。また、学習時に損失値を表示するなど、学習の進捗状況をモニタリングできるようにせよ。

import knock70
import knock71
import knock72
import torch
import torch.nn as nn
import torch.optim as optim

embedding = knock70.embeddings
train_data = knock71.train_data

model = knock72.NeuralNetworkModel(embedding)

criterion = nn.BCEWithLogitsLoss() #ニ値分類用の損失関数(引数はlogit, label)
optimizer = optim.SGD(model.parameters(), lr = 0.01) #最適化手法(引数はmodelのパラメータ, 学習率)

for epoch in range(10):
    
    total_loss = 0.0 #そのepochでの合計損失
    
    for example in train_data:
        
        input_ids = example["input_ids"]
        label = example["label"]
        
        optimizer.zero_grad() #勾配をリセット(Pytorchでは勾配が蓄積される)
        
        logit = model(input_ids) #テキストのID列から分類スコアを計算
        loss = criterion(logit, label) #分類スコア、正解ラベルから損失を計算
        
        loss.backward() #勾配を計算
        optimizer.step() #勾配を用いて(self.linearの)重みを更新
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_data) #損失の平均値
    print(epoch, "損失平均：", avg_loss)

torch.save(model.state_dict(), "model.pt") #学習結果(モデルの重み)を保存