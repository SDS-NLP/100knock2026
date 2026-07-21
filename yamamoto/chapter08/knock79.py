#ニューラルネットワークのアーキテクチャを自由に変更し、モデルを学習せよ。また、学習したモデルの開発セットにおける正解率を求めよ。例えば、テキストの特徴ベクトル（単語埋め込みの平均ベクトル）に対して多層のニューラルネットワークを通したり、畳み込みニューラルネットワーク（CNN; Convolutional Neural Network）や再帰型ニューラルネットワーク（RNN; Recurrent Neural Network）などのモデルの学習に挑戦するとよい。

import torch
import torch.nn as nn
import torch.optim as optim
import knock70
import knock71

class RecurrentNNModel(nn.Module): #RNNのモデルを設計
    
    def __init__(self, embeddings, hidden_size):
        
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embeddings, dtype = torch.float32),
            freeze = True,
            padding_idx = 0
        )
        
        self.rnn = nn.RNN(
            input_size = embeddings.shape[1], #単語ベクトルの次元数
            hidden_size = hidden_size, #記憶する単語ベクトルの次元数
            batch_first = True #入力テンソルの形を[batch_size, seq_len, embedding_dim]とする
        )
        
        self.linear = nn.Linear(hidden_size, 1) #隠れ層の次元数から1次元に線形変換
    
    def forward(self, input_ids):
        
       embeds = self.embedding(input_ids)
       output, hidden_n = self.rnn(embeds) #hidden_n:最後のRNN層の最後の隠れ状態
       
       last_hidden = hidden_n[-1]
       logit = self.linear(last_hidden)
       
       return logit

embeddings = knock70.embeddings

train_data = knock71.train_data
dev_data = knock71.dev_data

model = RecurrentNNModel(embeddings, hidden_size = 300)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(10):
    
    total_loss = 0.0
    
    for example in train_data:
        
        input_ids = example["input_ids"]
        labels = example["label"]
        
        optimizer.zero_grad()
        
        logit = model(input_ids)
        loss = criterion(logit, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_data)
    print(epoch, "損失平均：", avg_loss)

model.eval()

score = 0

with torch.no_grad():
    
    for example in dev_data:
        
        input_ids = example["input_ids"].unsqueeze(0)
        label = float(example["label"])
        
        logit = model(input_ids)
        
        prob = torch.sigmoid(logit)
        
        pred = (prob >= 0.5).float()
        
        if pred.item() == label:
            
            score += 1
    
    accuracy = score / len(dev_data)

print("正解率：", accuracy)