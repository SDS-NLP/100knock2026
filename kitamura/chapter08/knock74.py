import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class TextClassifier(nn.Module):
    def __init__(self, embedding_tensor):
        super().__init__() #親クラスの設定の引き継ぎ

        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor,
            padding_idx=0,
            freeze=True
        )
        embed_dim = embedding_tensor.size(1)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != 0).float().unsqueeze(-1)          
        sum_embeddings = (embedded * mask).sum(dim=1)  
        valid_lengths = mask.sum(dim=1).clamp(min=1)   
        avg_embedded = sum_embeddings / valid_lengths    
        out = self.linear(avg_embedded)
        return out



def calculate_accuracy(model, data_set):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in data_set:
            x = sample["input_ids"].unsqueeze(0)
            y = sample["label"].unsqueeze(0)
            outputs = model(x)
            predictions = (outputs >= 0.0).float()
            correct += (predictions == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


file1 = "word2vec_vocab_embedding.pt"
file2 = "sst_datasets.pt"

loaded_vocab = torch.load(file1)
loaded_datasets = torch.load(file2)
embedding_tensor = loaded_vocab["embedding_tensor"]

train_dataset = loaded_datasets["train"]
dev_dataset = loaded_datasets["dev"]


model = TextClassifier(embedding_tensor)
criterion = nn.BCEWithLogitsLoss()  #2値分類なのでbinary cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 10

for epoch in range(epochs):
    model.train()
    
    total_loss = 0.0
    
    for sample in train_dataset:
        input_ids = sample['input_ids']
        label = sample['label']
        x = input_ids.unsqueeze(0)
        y = label.unsqueeze(0)
        # 勾配のリセット
        optimizer.zero_grad()
        
        # 予測を出す
        outputs = model(x)
        
        # 損失を計算する
        loss = criterion(outputs, y)
        
        # 誤差逆伝播
        loss.backward()
        
        # 重みを更新する
        optimizer.step()
        
        # ログ表示用に損失を足しておく
        total_loss += loss.item()
        
        
train_acc = calculate_accuracy(model, train_dataset)
dev_acc = calculate_accuracy(model, dev_dataset)
print(train_acc)
print(dev_acc)

"""0.8389047261815454
0.786697247706422"""