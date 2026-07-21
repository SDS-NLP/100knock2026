import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_dataset(path, tokenizer):
    data = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            text, label = line.rstrip("\n").split("\t")
            enc = tokenizer(text, truncation=True, max_length=512)
            data.append({
                "label": torch.tensor(int(label)),          # ★int型に(理由は後述)
                "input_ids": torch.tensor(enc["input_ids"]),
            })
    return data

def collate(batch, pad_id=0):
    input_ids = [ex["input_ids"] for ex in batch]
    labels = torch.stack([ex["label"] for ex in batch])
    padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = (padded != pad_id).long()
    return {"input_ids": padded, "attention_mask": attention_mask, "labels": labels}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device) 

    train = load_dataset("SST-2/train.tsv", tokenizer)
    dev = load_dataset("SST-2/dev.tsv", tokenizer)
    train_loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev, batch_size=64, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)   # ★小さいlr

    def evaluate(loader):
        model.eval()
        n_correct, n_total = 0, 0
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits = model(**b).logits                   # (batch, 2)
                preds = logits.argmax(dim=-1)                # ★argmax(2クラス)
                n_correct += (preds == b["labels"]).sum().item()
                n_total += b["labels"].size(0)
        return n_correct / n_total

    best_acc = 0.0
    for epoch in range(1, 4):                                # ★3エポックで十分
        model.train()
        total_loss, n = 0.0, 0
        for step, b in enumerate(train_loader):
            b = {k: v.to(device) for k, v in b.items()}
            optimizer.zero_grad()
            out = model(**b)                                 # labelsを渡すと
            loss = out.loss                                  # lossまで計算済みで返る
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * b["labels"].size(0)
            n += b["labels"].size(0)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step:5d} | loss {loss.item():.4f}")

        dev_acc = evaluate(dev_loader)
        if dev_acc > best_acc:
            best_acc = dev_acc
            model.save_pretrained("knock87_best")            # transformers流の保存
        print(f"epoch {epoch} | train loss: {total_loss/n:.4f} | dev acc: {dev_acc:.4f}")

    print(f"best dev acc: {best_acc:.4f}")