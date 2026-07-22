import os

import pandas as pd
from transformers import AutoTokenizer


def load_dataset(path, tokenizer):
    df = pd.read_csv(path, sep="\t")
    texts = df["sentence"].tolist()
    labels = df["label"].tolist()
    tokens = [tokenizer.tokenize(text) for text in texts]

    return df, texts, labels, tokens

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_dir = os.path.join(os.path.dirname(__file__), "data", "SST-2")
    train_df, train_texts, train_labels, train_tokens = load_dataset(
        os.path.join(data_dir, "train.tsv"), tokenizer
    )
    dev_df, dev_texts, dev_labels, dev_tokens = load_dataset(
        os.path.join(data_dir, "dev.tsv"), tokenizer
    )
    print(f"train_tokens: {train_tokens[0]} train_label: {train_labels[0]}")
    print(f"元データ(train): {train_df['sentence'].iloc[0]}")
    print("-"*60)
    print(f"dev_tokens: {dev_tokens[0]} dev_label: {dev_labels[0]}")
    print(f"元データ(dev): {dev_df['sentence'].iloc[0]}")

if __name__ == "__main__":
    main()
