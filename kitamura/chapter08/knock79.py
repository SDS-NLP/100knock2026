import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F

class DeepTextClassifier(nn.Module):
    def __init__(self, embedding_tensor, hidden_dim=128, drop_rate=0.5):
        super().__init__() #親クラスの設定の引き継ぎ

        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor,
            padding_idx=0,
            freeze=True
        )
        embed_dim = embedding_tensor.size(1)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)  # 128次元に圧縮
        self.dropout = nn.Dropout(drop_rate)  # 過学習を防ぐために50%の確率でシャットダウン
        self.linear2 = nn.linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != 0).float().unsqueeze(-1)          
        sum_embeddings = (embedded * mask).sum(dim=1)  
        valid_lengths = mask.sum(dim=1).clamp(min=1)   
        avg_embedded = sum_embeddings / valid_lengths    
        out = self.linear1(avg_embedded)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out


def collate_fn(batch):
    input_ids = []
    labels = []
    
    for item in batch:
        input_ids.append(item["input_ids"])
        labels.append(item["label"])
    
    padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return {'input_ids': padded_ids, 'labels': labels}


def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch["input_ids"]
            y = batch["labels"]
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


batch_size = 32
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
)

dev_loader = DataLoader(
    dev_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
)

model = DeepTextClassifier(embedding_tensor)
criterion = nn.BCEWithLogitsLoss()   #2値分類なのでbinary cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 10

for epoch in range(epochs):
    model.train()
    
    total_loss = 0.0
    
    # train_loaderから32件ずつデータを取り出す
    for batch in train_loader:
        x = batch['input_ids']
        y = batch['labels']
        
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
        
        
train_acc = calculate_accuracy(model, train_loader)
dev_acc = calculate_accuracy(model, dev_loader)
print(train_acc)
print(dev_acc)