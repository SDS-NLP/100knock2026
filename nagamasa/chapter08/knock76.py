import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from knock70 import build_embedding_matrix
from knock71 import DEV_PATH, TRAIN_PATH, build_dataset, load_sst2
from knock75 import collate

# 76. ミニバッチ学習
# 75 の collate + DataLoader でバッチ化して学習し、dev 正解率を出す。
# 72 のモデルは (L,) 単一系列前提で 73/74 が依存しているので壊さず、
# ここで (B, L_max) を受けるバッチ版 BoWClassifier を別に用意する。
# 文ベクトル = パディングを除いた実トークンだけの平均(マスク平均)。

BATCH_SIZE = 64
LR = 0.01
EPOCHS = 10


class BoWClassifier(nn.Module):
    def __init__(self, E):
        super().__init__()
        # freeze=True: 学習中は埋め込みを固定(73 の指示)。padding_idx=0: PAD 行は常にゼロ。
        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(E), freeze=True, padding_idx=0
        )
        self.fc = nn.Linear(self.emb.embedding_dim, 1)  # 300 → 1

    def forward(self, input_ids):
        # input_ids: (B, L_max) long
        vecs = self.emb(input_ids)                 # (B, L_max, 300)
        # マスク平均: 和は PAD(ゼロ行)を足しても不変。割る数だけを実トークン数にする。
        mask = input_ids != 0                      # (B, L_max) 実=True, PAD=False
        lengths = mask.sum(dim=1, keepdim=True)    # (B, 1) 各文の実トークン数
        summed = vecs.sum(dim=1)                   # (B, 300) トークン軸の和(PADは0で無害)
        feat = summed / lengths                    # (B, 300) 実長で割る = 正しい平均
        return torch.sigmoid(self.fc(feat))        # (B, 1) ポジティブ確率


def train_model(model, train_data, batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS):
    # collate_fn に 75 の collate を渡す。shuffle=True で毎エポック並べ替え。
    loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    criterion = nn.BCELoss()  # sigmoid 済みの確率 → BCELoss
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            prob = model(batch["input_ids"])        # (B, 1)
            loss = criterion(prob, batch["label"])  # (B,1) vs (B,1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # BCELoss は既定でバッチ平均。事例数で重み付けして足し、エポック平均に戻す。
            total_loss += loss.item() * batch["label"].size(0)
        print(f"epoch {epoch}: loss = {total_loss / len(train_data):.4f}")
    return model


def accuracy(model, data, batch_size=BATCH_SIZE):
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            prob = model(batch["input_ids"])          # (B, 1)
            pred = (prob >= 0.5).float()              # (B, 1) 0/1
            # (B,1) 同士の要素比較 → 一致数。75 で label を stack にしたのがここで効く。
            correct += (pred == batch["label"]).sum().item()
    return correct / len(data)


if __name__ == "__main__":
    E, word2id, id2word = build_embedding_matrix()
    train = build_dataset(load_sst2(TRAIN_PATH), word2id)
    dev = build_dataset(load_sst2(DEV_PATH), word2id)

    model = BoWClassifier(E)
    train_model(model, train)
    print("dev accuracy:", accuracy(model, dev))
