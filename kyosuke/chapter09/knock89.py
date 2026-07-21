import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel


def load_dataset(path, tokenizer):
    data = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            text, label = line.rstrip("\n").split("\t")
            enc = tokenizer(text, truncation=True, max_length=512)
            data.append({
                "label": torch.tensor(int(label)),
                "input_ids": torch.tensor(enc["input_ids"]),
            })
    return data


def collate(batch, pad_id=0):
    input_ids = [ex["input_ids"] for ex in batch]
    labels = torch.stack([ex["label"] for ex in batch])
    padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = (padded != pad_id).long()
    return {"input_ids": padded, "attention_mask": attention_mask, "labels": labels}


class BertPoolingClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", pooling="max", num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)     # ヘッドなしの素のBERT(83,84と同じ)
        self.pooling = pooling
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # 768 → 2


    def forward(self, input_ids, attention_mask):
        last_hidden = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                                   # (batch, seq_len, 768)

        if self.pooling == "cls":
            feat = last_hidden[:, 0, :]                       # 83方式
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)               # 84方式(マスク平均)
            feat = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        elif self.pooling == "max":                           # ★今回の本命
            mask = attention_mask.unsqueeze(-1).bool()
            feat = last_hidden.masked_fill(~mask, float("-inf")).max(dim=1).values
        return self.classifier(feat)                          # (batch, 2) ロジット


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train = load_dataset("SST-2/train.tsv", tokenizer)
    dev = load_dataset("SST-2/dev.tsv", tokenizer)
    train_loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev, batch_size=64, shuffle=False, collate_fn=collate)

    model = BertPoolingClassifier(pooling="max").to(device)
    criterion = nn.CrossEntropyLoss()                         # 損失は自分で用意(8章スタイル復帰)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def evaluate(loader):
        model.eval()
        n_correct, n_total = 0, 0
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits = model(b["input_ids"], b["attention_mask"])
                preds = logits.argmax(dim=-1)
                n_correct += (preds == b["labels"]).sum().item()
                n_total += b["labels"].size(0)
        return n_correct / n_total

    best_acc = 0.0
    for epoch in range(1, 4):
        model.train()
        total_loss, n = 0.0, 0
        for step, b in enumerate(train_loader):
            b = {k: v.to(device) for k, v in b.items()}
            optimizer.zero_grad()
            logits = model(b["input_ids"], b["attention_mask"])
            loss = criterion(logits, b["labels"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * b["labels"].size(0)
            n += b["labels"].size(0)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step:5d} | loss {loss.item():.4f}")

        dev_acc = evaluate(dev_loader)
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(), "knock89_best.pt")
        print(f"epoch {epoch} | train loss: {total_loss/n:.4f} | dev acc: {dev_acc:.4f}")

    print(f"best dev acc ({model.pooling} pooling): {best_acc:.4f}")