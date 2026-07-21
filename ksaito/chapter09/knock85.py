import pandas as pd
from transformers import AutoTokenizer


def load_dataset(path, tokenizer):
    df = pd.read_csv(path, sep="\t")
    texts = df["sentence"].tolist()
    labels = df["label"].tolist()
    tokens = [tokenizer.tokenize(text) for text in texts]

    return texts, labels, tokens

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_texts, train_labels, train_tokens = load_dataset(
        "data/SST-2/train.tsv", tokenizer
    )
    dev_texts, dev_labels, dev_tokens = load_dataset(
        "data/SST-2/dev.tsv", tokenizer
    )
    print(f"train_tokens: {train_tokens[0]} train_label: {train_labels[0]}")
    print(f"dev_tokens: {dev_tokens[0]} dev_label: {dev_labels[0]}")

if __name__ == "__main__":
    main()
