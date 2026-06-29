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
        
    # 1エポック終わるごとに、平均の損失を表示して進捗をモニタリング
    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{epochs}] | 損失(Loss): {avg_loss:.4f}")

print("学習が完了しました")

"""Epoch [1/10] | 損失(Loss): 0.3884
Epoch [2/10] | 損失(Loss): 0.3767
Epoch [3/10] | 損失(Loss): 0.3763
Epoch [4/10] | 損失(Loss): 0.3762
Epoch [5/10] | 損失(Loss): 0.3762
Epoch [6/10] | 損失(Loss): 0.3762
Epoch [7/10] | 損失(Loss): 0.3762
Epoch [8/10] | 損失(Loss): 0.3762
Epoch [9/10] | 損失(Loss): 0.3762
Epoch [10/10] | 損失(Loss): 0.3762"""