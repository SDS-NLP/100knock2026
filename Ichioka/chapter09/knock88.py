"""
SST-2 データで事前学習済みBERTをファインチューニングし、
指定した5つの文の極性（ポジティブ/ネガティブ）を予測するスクリプト

やること:
    1. train.tsv / dev.tsv を読み込む
    2. BertForSequenceClassification をファインチューニングする
    3. 検証セットで正解率を確認する
    4. 指定した5文についてファインチューニング済みモデルで極性を予測する
"""

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW


# ===== ハイパーパラメータ =====
MODEL_NAME = "bert-base-uncased"   # 事前学習済みモデル名
MAX_LENGTH = 64                    # トークン列の最大長
BATCH_SIZE = 16
NUM_EPOCHS = 1                     # 動作確認用。精度を上げたい場合は増やす
LEARNING_RATE = 2e-5
TRAIN_PATH = "SST-2/SST-2/train.tsv"
DEV_PATH = "SST-2/SST-2/dev.tsv"

# 極性を予測したい文（この課題で指定された5文）
TARGET_SENTENCES = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

# ラベルIDと人間が読みやすい名前の対応（SST-2は 0=negative, 1=positive）
ID2LABEL = {0: "negative", 1: "positive"}


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

    # 1行目がヘッダーかどうかを簡易判定（ラベル列が数値でなければヘッダーとみなす）
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


def train_one_epoch(model, data_loader, optimizer, device):
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def predict_sentences(model, tokenizer, sentences, device, max_length):
    """
    ファインチューニング済みモデルを用いて、文のリストの極性を予測する

    戻り値:
        results: [{"text": str, "label": str, "prob": float}, ...]
    """
    model.eval()

    # 予測したい文をまとめてトークン化・パディングする
    encoded = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # logitsをsoftmaxで確率に変換する
        probs = F.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    results = []
    for i, sentence in enumerate(sentences):
        pred_id = preds[i].item()
        pred_prob = probs[i, pred_id].item()
        results.append({
            "text": sentence,
            "label": ID2LABEL[pred_id],
            "prob": pred_prob,
        })

    return results


def main():
    # GPUが使える場合はGPUを使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # トークナイザとモデルを読み込む
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # データを読み込む
    train_data = load_tsv(TRAIN_PATH)
    dev_data = load_tsv(DEV_PATH)
    print(f"訓練セット件数: {len(train_data)}")
    print(f"検証セット件数: {len(dev_data)}")

    # Dataset / DataLoader を構築する
    collate_fn = make_collate_fn(tokenizer, MAX_LENGTH)
    train_loader = DataLoader(
        SST2Dataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        SST2Dataset(dev_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # オプティマイザを設定する
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # ファインチューニングを実行する
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        accuracy = evaluate(model, dev_loader, device)
        print(f"[Epoch {epoch}] 平均損失: {avg_loss:.4f} / 検証セット正解率: {accuracy:.4f}")

    # ===== 指定された5文の極性を予測する =====
    results = predict_sentences(model, tokenizer, TARGET_SENTENCES, device, MAX_LENGTH)

    print("\n--- 予測結果 ---")
    for r in results:
        print(f"文     : {r['text']}")
        print(f"予測極性: {r['label']} (確信度: {r['prob']:.4f})")
        print("-" * 40)


if __name__ == "__main__":
    main()