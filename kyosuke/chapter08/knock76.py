import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from knock74 import load_dataset,E

def collate(batch):
    batch = sorted(batch, key=lambda ex: len(ex["input_ids"]), reverse=True)
    input_ids = [ex["input_ids"] for ex in batch]
    labels = torch.stack([ex["label"] for ex in batch])
    padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(x) for x in input_ids])  # マスク平均用
    return {"input_ids": padded, "label": labels, "lengths": lengths}

# ---------- モデル(埋め込み固定・マスク平均) ----------
class BoWClassifier(nn.Module):
    def __init__(self, embedding_weights, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=True, padding_idx=padding_idx
        )
        self.fc = nn.Linear(embedding_weights.size(1), 1)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)          # (batch, max_len, 300)
        feat = emb.sum(dim=1) / lengths.unsqueeze(1)  # PADを除いた平均
        return self.fc(feat)                     # (batch, 1)

# ---------- DataLoader:ミニバッチの供給係 ----------
train = load_dataset("SST-2/train.tsv")
dev = load_dataset("SST-2/dev.tsv")

batch_size = 64
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,  collate_fn=collate)
dev_loader   = DataLoader(dev,   batch_size=batch_size, shuffle=False, collate_fn=collate)

# ---------- 学習 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BoWClassifier(E).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(loader):
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            lens = b["lengths"].to(device)
            labels = b["label"].to(device)
            preds = (torch.sigmoid(model(ids, lens)) >= 0.5).float()
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)
    return n_correct / n_total

for epoch in range(1, 11):
    model.train()
    total_loss, n = 0.0, 0
    for b in train_loader:                       # ← ミニバッチが1個ずつ流れてくる
        ids = b["input_ids"].to(device)
        lens = b["lengths"].to(device)
        labels = b["label"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(ids, lens), labels)
        loss.backward()                          # ミニバッチ64件分の勾配を計算
        optimizer.step()                         # 重みを1回更新

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

    dev_acc = evaluate(dev_loader)
    print(f"epoch {epoch:2d} | train loss: {total_loss/n:.4f} | dev acc: {dev_acc:.4f}")