import pandas as pd
from transformers import AutoTokenizer


def main():
    train = pd.read_csv("data/SST-2/train.tsv", sep="\t")
    dev = pd.read_csv("data/SST-2/dev.tsv", sep="\t")

    train_text = train["sentence"].tolist()
    train_label = train["label"].tolist()

    dev_text = dev["sentence"].tolist()
    dev_label = dev["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_batch = tokenizer(train_text[:4], padding=True, truncation=True, return_tensors="pt")

    dev_batch = tokenizer(dev_text[:4], padding=True, truncation=True, return_tensors="pt")

    print("train: input_ids")
    print(train_batch["input_ids"])

    print("\nattention_mask")
    print(train_batch["attention_mask"])

    print("\nlabels")
    print(train_label[:4])

    print("dev: input_ids")
    print(dev_batch["input_ids"])

    print("\nattention_mask")
    print(dev_batch["attention_mask"])

    print("\nlabels")
    print(dev_label[:4])


if __name__ == "__main__":
    main()
