import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common import build_embedding, load_dataset, collate, BoWClassifier

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    E, token2id = build_embedding(limit=500000)          # 語彙制限
    train = load_dataset("SST-2/train.tsv", token2id)
    dev = load_dataset("SST-2/dev.tsv", token2id)

    train_loader = DataLoader(train, batch_size=64, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev, batch_size=64, shuffle=False, collate_fn=collate)

    model = BoWClassifier(E, freeze=False).to(device)    # ★78問目の本体
    print("embedding requires_grad:", model.embedding.weight.requires_grad)  # True

    before = model.embedding.weight.detach().clone()     # 検証用に学習前の状態を保存

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def evaluate(loader):
        model.eval()
        n_correct, n_total = 0, 0
        with torch.no_grad():
            for b in loader:
                ids, lens, labels = b["input_ids"].to(device), b["lengths"].to(device), b["label"].to(device)
                preds = (torch.sigmoid(model(ids, lens)) >= 0.5).float()
                n_correct += (preds == labels).sum().item()
                n_total += labels.size(0)
        return n_correct / n_total

    best_acc = 0.0
    for epoch in range(1, 11):
        model.train()
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
        best_acc = max(best_acc, dev_acc)
        print(f"epoch {epoch:2d} | train loss: {total_loss/n:.4f} | dev acc: {dev_acc:.4f}")

    print(f"best dev acc: {best_acc:.4f}")