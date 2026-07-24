import csv
from pathlib import Path

from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
# The SST-2 (GLUE) files were downloaded for chapter08; reuse them in place.
DATA_DIR = Path(__file__).resolve().parent.parent / "chapter08" / "SST-2"


def load_split(path):
    """Read one GLUE SST-2 .tsv file into parallel lists of texts and labels.

    The file starts with a `sentence\tlabel` header; labels are 0 (negative) or
    1 (positive).
    """
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            texts.append(row[0].strip())
            labels.append(int(row[1]))
    return texts, labels


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_texts, train_labels = load_split(DATA_DIR / "train.tsv")
    dev_texts, dev_labels = load_split(DATA_DIR / "dev.tsv")

    # Convert every text into a WordPiece token sequence.
    train_tokens = [tokenizer.tokenize(t) for t in train_texts]
    dev_tokens = [tokenizer.tokenize(t) for t in dev_texts]

    print(f"train: {len(train_texts)} examples")
    print(f"dev:   {len(dev_texts)} examples")

    print("\n-- first training example --")
    print("text  :", train_texts[0])
    print("label :", train_labels[0])
    print("tokens:", train_tokens[0])

    print("\n-- first dev example --")
    print("text  :", dev_texts[0])
    print("label :", dev_labels[0])
    print("tokens:", dev_tokens[0])


if __name__ == "__main__":
    main()
