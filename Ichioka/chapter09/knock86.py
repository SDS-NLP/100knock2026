"""
SST-2 データからミニバッチを構成するスクリプト

やること:
    1. train.tsv を読み込む
    2. 冒頭4事例を取り出す
    3. transformers の BertTokenizer でトークン化・ID化する
    4. 系列長を揃える（パディング）
    5. ミニバッチ（テンソル）を構成する
"""

import csv
from pathlib import Path

import torch
from transformers import BertTokenizer


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


def build_batch(texts, labels, tokenizer, max_length=None):
    """
    テキストのリストをBERTトークナイザでID化し、パディングしてミニバッチを作る

    引数:
        texts: 文字列のリスト
        labels: ラベル（int）のリスト
        tokenizer: BertTokenizer
        max_length: 系列長の上限（Noneの場合はバッチ内最大長に合わせる）

    戻り値:
        dict: input_ids, attention_mask, labels のテンソルをまとめたもの
    """
    # tokenizer自体にパディング機能があるので、それを利用する
    # padding="longest" -> バッチ内で最も長い系列に合わせてパディング
    # padding="max_length" と max_length を指定すれば固定長にもできる
    if max_length is not None:
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

    batch = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    return batch


def main():
    # BERTトークナイザを読み込む（英語版・小文字化モデルを例として使用）
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 訓練データを読み込む
    train_data = load_tsv("SST-2/SST-2/train.tsv")

    # 冒頭4事例を取り出す
    samples = train_data[:4]
    texts = [text for text, label in samples]
    labels = [label for text, label in samples]

    print("--- 元のテキスト ---")
    for text, label in samples:
        print(f"ラベル={label} テキスト={text}")

    # トークン化した結果（パディング前）も確認してみる
    print("\n--- トークン化結果（パディング前） ---")
    for text in texts:
        tokens = tokenizer.tokenize(text)
        print(f"トークン数={len(tokens)} トークン列={tokens}")

    # ミニバッチを構成する（バッチ内最大長に合わせてパディング）
    batch = build_batch(texts, labels, tokenizer)

    print("\n--- ミニバッチ（パディング後） ---")
    print(f"input_ids の形状     : {batch['input_ids'].shape}")
    print(f"attention_mask の形状: {batch['attention_mask'].shape}")
    print(f"labels              : {batch['labels']}")

    print("\ninput_ids:")
    print(batch["input_ids"])
    print("\nattention_mask:")
    print(batch["attention_mask"])


if __name__ == "__main__":
    main()