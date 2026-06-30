import csv
from collections import Counter

DATA_DIR = "./SST-2"

def load_dataset(filepath):
    dataset = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["sentence"].strip()
            label = row["label"]
            feature = Counter(text.split())
            dataset.append({"text": text, "label": label, "feature": dict(feature)})
    return dataset

train = load_dataset(f"{DATA_DIR}/train.tsv")
dev = load_dataset(f"{DATA_DIR}/dev.tsv")

print(f"train size: {len(train)}, dev size: {len(dev)}")
print("train[0]:", train[0])
