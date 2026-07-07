import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary=True
)
token2id = {"<PAD>": 0}
token2id.update({tok: i + 1 for i, tok in enumerate(wv.index_to_key)})
V, d_emb = len(wv.index_to_key) + 1, wv.vector_size

E = np.zeros((V, d_emb), dtype=np.float32)
E[1:] = wv.vectors
E = torch.from_numpy(E)

def load_dataset(path):
    data = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            text, label = line.rstrip("\n").split("\t")
            ids = [token2id[t] for t in text.split() if t in token2id]
            if not ids:
                continue
            data.append({
                "text": text,
                "label": torch.tensor([float(label)]),
                "input_ids": torch.tensor(ids),
            })
    return data

train = load_dataset("SST-2/train.tsv")

def collate_fn(batch):
    ids = [ex["input_ids"] for ex in batch]
    labels = torch.stack([ex["label"] for ex in batch])
    padded = pad_sequence(ids, batch_first=True, padding_value=0)  # (batch, max_len)
    lengths = torch.tensor([len(x) for x in ids])                  # 有効トークン数
    return padded, lengths, labels

class BoWClassifier(nn.Module):
    def __init__(self, embedding_weights, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=True, padding_idx=padding_idx  # ★固定
        )
        self.fc = nn.Linear(embedding_weights.size(1), 1)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        summed = emb.sum(dim=1)
        feat = summed / lengths.unsqueeze(1)
        return self.fc(feat)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BoWClassifier(E).to(device)

loader = DataLoader(train, batch_size=64, shuffle=True, collate_fn=collate_fn)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 10
for epoch in range(1, n_epochs + 1):
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0
    for padded, lengths, labels in loader:
        padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(padded, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        n_correct += (preds == labels).sum().item()
        n_total += labels.size(0)

    avg_loss = total_loss / n_total
    acc = n_correct / n_total
    print(f"epoch {epoch:2d} | loss: {avg_loss:.4f} | train acc: {acc:.4f}")

torch.save(model.state_dict(), "knock73_model.pt")