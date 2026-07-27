"""
86. ミニバッチの作成
85で読み込んだ訓練データの一部（例えば冒頭の4事例）に対して、パディングなどの処理を行い、トークン列の長さを揃えてミニバッチを構成せよ。
"""

import pandas as pd
from transformers import AutoTokenizer
import torch

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df["sentence"].tolist(), df["label"].tolist()


def tokenize_texts(texts):
    tokenized_texts = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        tokenized_texts.append(tokens)
    return tokenized_texts


train_path = "../chapter08/SST-2/train.tsv"
dev_path = "../chapter08/SST-2/dev.tsv"

train_texts, train_labels = load_data(train_path)
dev_texts, dev_labels = load_data(dev_path)

train_tokenized = tokenize_texts(train_texts)
dev_tokenized = tokenize_texts(dev_texts)

sample_texts = train_texts[:4]
sample_labels = train_labels[:4]
sample_tokenized = train_tokenized[:4]

encoded = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

print("元のテキスト:")
for text in sample_texts:
    print(f"- {text}")

print("\nトークン列:")
for tokens in sample_tokenized:
    print(f"- {tokens}")

print("\nパディング後のトークンID:")
print(encoded["input_ids"])

print("\nアテンションマスク:")
print(encoded["attention_mask"])

print("\nラベル:")
print(torch.tensor(sample_labels))