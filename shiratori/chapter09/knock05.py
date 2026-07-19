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

    train_tokens = [tokenizer.tokenize(sentence) for sentence in train_text]
    dev_tokens = [tokenizer.tokenize(sentence) for sentence in dev_text]

    print("Train:")
    print(train_tokens[0])
    print("Label:", train_label[0])

    print("\nDev:")
    print(dev_tokens[0])
    print("Label:", dev_label[0])


if __name__ == "__main__":
    main()


# Train:
# ['hide', 'new', 'secret', '##ions', 'from', 'the', 'parental', 'units']
# Label: 0

# Dev:
# ['it', "'", 's', 'a', 'charming', 'and', 'often', 'affecting', 'journey', '.']
# Label: 1
