#問題75のパディングの処理を活用して、ミニバッチでモデルを学習せよ。また、学習したモデルの開発セットにおける正解率を求めよ。

import knock70
import knock71
import knock72
import knock75
import torch
import torch.nn as nn
import torch.optim as optim

embedding = knock70.embeddings
train_data = knock71.train_data
dev_data = knock71.dev_data

train_batch = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = 32,
    shuffle = True,
    collate_fn = knock75.collate
)

model = knock72.NeuralNetworkModel(embedding)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(10):
    
    total_loss = 0.0
    
    for batch in train_batch:
        
        input_ids = batch["input_ids"]
        labels = batch["label"]
        
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

with torch.no_grad():
    
    for example in dev_data:
    
        input_ids = example["input_ids"]
        label = float(example["label"])
        
        logit = model(input_ids.unsqueeze(0)) #モデルで分類スコアを計算
        
        prob = torch.sigmoid(logit) #シグモイド関数で確率に変換
        
        pred = (prob >= 0.5).float() #0.5以上で1.0(True), それ以下は0.0(False)でラベルを予測
        
        if pred.item() == label:
            
            score += 1
    
    accuracy = score / len(dev_data)

print("正解率：", accuracy)