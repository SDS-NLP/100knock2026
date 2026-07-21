import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from knock74 import load_dataset, E
from knock75 import collate

class MLPClassifier(nn.Module):
    def __init__(self, embedding_weights, hidden_dim=256, dropout=0.3,
                 padding_idx=0, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=freeze, padding_idx=padding_idx
        )
        d_emb = embedding_weights.size(1)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, hidden_dim),   # 300 → 256
            nn.ReLU(),                      # 非線形(これが本体)
            nn.Dropout(dropout),            # 過学習対策
            nn.Linear(hidden_dim, 1),       # 256 → 1 (ロジット)
        )

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        feat = emb.sum(dim=1) / lengths.unsqueeze(1)   # マスク平均(従来通り)
        return self.mlp(feat)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train = load_dataset("SST-2/train.tsv")
    dev = load_dataset("SST-2/dev.tsv")
    train_loader = DataLoader(train, batch_size=64, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev, batch_size=64, shuffle=False, collate_fn=collate)

    model = MLPClassifier(E, hidden_dim=256, dropout=0.3, freeze=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def evaluate(loader):
        model.eval()                      # ★Dropoutが入った今、これが実質的な意味を持つ
        n_correct, n_total = 0, 0
        with torch.no_grad():
            for b in loader:
                ids, lens, labels = b["input_ids"].to(device), b["lengths"].to(device), b["label"].to(device)
                preds = (torch.sigmoid(model(ids, lens)) >= 0.5).float()
                n_correct += (preds == labels).sum().item()
                n_total += labels.size(0)
        return n_correct / n_total

    best_acc = 0.0
    for epoch in range(1, 16):
        model.train()                     # ★Dropoutを有効にする側
        total_loss, n = 0.0, 0
        for b in train_loader:
            ids, lens, labels = b["input_ids"].to(device), b["lengths"].to(device), b["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, lens), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)

        dev_acc = evaluate(dev_loader)
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(), "knock79_best.pt")   # 最良時点を保存
        print(f"epoch {epoch:2d} | train loss: {total_loss/n:.4f} | dev acc: {dev_acc:.4f}")

    print(f"best dev acc: {best_acc:.4f}")