"""
問題87とは異なるアーキテクチャで極性分析モデルを設計・ファインチューニングするスクリプト

"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW


# ===== ハイパーパラメータ =====
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 16
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
NUM_LABELS = 2                     # SST-2: 0=negative, 1=positive
TRAIN_PATH = "SST-2/SST-2/train.tsv"
DEV_PATH = "SST-2/SST-2/dev.tsv"


def load_tsv(file_path: str):
    """
    tsvファイルを読み込み、(テキスト, ラベル) のリストを返す
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    # 1行目がヘッダーかどうかを簡易判定
    start_idx = 0
    if rows and len(rows[0]) >= 2 and not rows[0][1].strip().isdigit():
        start_idx = 1

    data = []
    for row in rows[start_idx:]:
        if len(row) < 2:
            continue
        text, label = row[0], row[1]
        data.append((text, int(label)))

    return data


class SST2Dataset(Dataset):
    """
    (テキスト, ラベル) のリストを保持するだけのシンプルなDatasetクラス
    """

    def __init__(self, data):
        self.texts = [text for text, label in data]
        self.labels = [label for text, label in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_collate_fn(tokenizer, max_length):
    """
    バッチ内のテキストをまとめてトークン化・パディングするcollate_fnを返す
    """

    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


class MaxPoolingClassifier(nn.Module):
    """
    [CLS]トークンを使わず、全トークンの埋め込みに対する最大値プーリングを用いる分類モデル

    構造:
        BertModel(入力) -> 各トークンの埋め込み (batch, seq_len, hidden_size)
        -> パディング部分をマスクした上で、系列方向(seq_len)に沿って最大値プーリング
        -> (batch, hidden_size) のベクトルを線形層に入力してクラス数に変換
    """

    def __init__(self, model_name, num_labels, dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # BertModelの出力から、各トークンの埋め込み（最終層の隠れ状態）を取得する
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # パディング部分が最大値プーリングに影響しないよう、
        # attention_maskが0の位置を非常に小さい値(-inf相当)に置き換えてからmaxを取る
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())  # (batch, seq_len, hidden_size)
        masked_embeddings = token_embeddings.masked_fill(mask == 0, -1e9)

        # 系列長方向(dim=1)に沿って最大値を取る -> (batch, hidden_size)
        pooled, _ = torch.max(masked_embeddings, dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, num_labels)
        return logits


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """
    1エポック分の学習を行う
    """
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """
    検証セット上での正解率（accuracy）を計算する
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 訓練・検証データを読み込む
    train_data = load_tsv(TRAIN_PATH)
    dev_data = load_tsv(DEV_PATH)
    print(f"訓練セット件数: {len(train_data)}")
    print(f"検証セット件数: {len(dev_data)}")

    collate_fn = make_collate_fn(tokenizer, MAX_LENGTH)
    train_loader = DataLoader(
        SST2Dataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        SST2Dataset(dev_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # 最大値プーリングを用いる独自の分類モデルを構築する
    model = MaxPoolingClassifier(MODEL_NAME, NUM_LABELS)
    model.to(device)

    # BertForSequenceClassificationと違い、labelsを渡しても自動でlossを計算してくれないため、
    # 損失関数（交差エントロピー）を自前で用意する
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        accuracy = evaluate(model, dev_loader, device)
        print(f"[Epoch {epoch}] 平均損失: {avg_loss:.4f} / 検証セット正解率: {accuracy:.4f}")

    final_accuracy = evaluate(model, dev_loader, device)
    print(f"\n最終的な検証セット正解率（最大値プーリングモデル）: {final_accuracy:.4f}")


if __name__ == "__main__":
    main()