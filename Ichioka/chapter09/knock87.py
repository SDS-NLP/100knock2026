"""
SST-2 データを用いて事前学習済みBERTを極性分析タスクにファインチューニングし、
検証セット（dev.tsv）上での正解率を計算するスクリプト

やること:
    1. train.tsv / dev.tsv を読み込む
    2. BertTokenizer でトークン化・パディングし、DataLoaderを作る
    3. BertForSequenceClassification をファインチューニングする
    4. 検証セット上での正解率（accuracy）を計算する
"""

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW


# ===== ハイパーパラメータ =====
MODEL_NAME = "bert-base-uncased"   # 事前学習済みモデル名
MAX_LENGTH = 64                    # トークン列の最大長（これに合わせてパディング/切り詰め）
BATCH_SIZE = 16
NUM_EPOCHS = 1                     # 動作確認用に1エポックにしているが、必要に応じて増やす
LEARNING_RATE = 2e-5
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
    (テキスト, ラベル) のリストをBERT用のテンソルに変換するDatasetクラス
    """

    def __init__(self, data, tokenizer, max_length):
        self.texts = [text for text, label in data]
        self.labels = [label for text, label in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 1件ずつトークン化する（バッチ化・パディングはDataLoaderのcollate_fnで行う）
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


def make_collate_fn(tokenizer, max_length):
    """
    DataLoaderのバッチ生成時に、テキストをまとめてトークン化・パディングする関数を返す
    """

    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        encoded = tokenizer(
            texts,
            padding="max_length",   # 固定長にパディング（バッチごとに長さが変わらないようにする）
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

        # BertForSequenceClassification は labels を渡すと自動でlossを計算してくれる
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate(model, data_loader, device):
    """
    検証セット上での正解率（accuracy）を計算する
    """
    model.eval()
    correct = 0
    total = 0

    # 評価時は勾配計算が不要なので torch.no_grad() で囲む
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 最もスコアの高いクラスを予測ラベルとする
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def main():
    # GPUが使える場合はGPUを使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # トークナイザとモデルを読み込む
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # num_labels=2 -> 極性分析（ポジティブ/ネガティブの2クラス分類）
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # データを読み込む
    train_data = load_tsv(TRAIN_PATH)
    dev_data = load_tsv(DEV_PATH)
    print(f"訓練セット件数: {len(train_data)}")
    print(f"検証セット件数: {len(dev_data)}")

    # Dataset / DataLoader を構築する
    train_dataset = SST2Dataset(train_data, tokenizer, MAX_LENGTH)
    dev_dataset = SST2Dataset(dev_data, tokenizer, MAX_LENGTH)

    collate_fn = make_collate_fn(tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # オプティマイザを設定する
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # ファインチューニングを実行する
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch}] 平均損失: {avg_loss:.4f}")

        # 各エポック終了時に検証セットで評価する
        accuracy = evaluate(model, dev_loader, device)
        print(f"[Epoch {epoch}] 検証セット正解率: {accuracy:.4f}")

    # 最終的な正解率を改めて表示する
    final_accuracy = evaluate(model, dev_loader, device)
    print(f"\n最終的な検証セット正解率: {final_accuracy:.4f}")


if __name__ == "__main__":
    main()