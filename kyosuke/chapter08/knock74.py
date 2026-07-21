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

def collate_fn(batch):
    ids = [ex["input_ids"] for ex in batch]
    labels = torch.stack([ex["label"] for ex in batch])
    padded = pad_sequence(ids, batch_first=True, padding_value=0)  # (batch, max_len)
    lengths = torch.tensor([len(x) for x in ids])                  # 有効トークン数
    return padded, lengths, labels

class BoWClassifier(nn.Module):
    def __init__(self, embedding_weights, padding_idx=0,freeze=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=freeze, padding_idx=padding_idx  # ★固定
        )
        self.fc = nn.Linear(embedding_weights.size(1), 1)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        summed = emb.sum(dim=1)
        feat = summed / lengths.unsqueeze(1)
        return self.fc(feat)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoWClassifier(E).to(device)
    model.load_state_dict(torch.load("knock73_model.pt", map_location=device))

    dev = load_dataset("SST-2/dev.tsv")
    dev_loader = DataLoader(dev, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for padded, lengths, labels in dev_loader:
            padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
            logits = model(padded, lengths)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)

    print(f"dev accuracy: {n_correct / n_total:.4f}  ({n_correct}/{n_total})")