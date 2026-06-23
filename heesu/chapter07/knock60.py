import csv
from collections import Counter

DATA_DIR = "./SST-2"

def count_labels(filepath):
    counts = Counter()
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            counts[row["label"]] += 1
    return counts

train_counts = count_labels(f"{DATA_DIR}/train.tsv")
dev_counts = count_labels(f"{DATA_DIR}/dev.tsv")

print("train.tsv:")
print(f"  Positive (1): {train_counts['1']}")
print(f"  Negative (0): {train_counts['0']}")

print("dev.tsv:")
print(f"  Positive (1): {dev_counts['1']}")
print(f"  Negative (0): {dev_counts['0']}")
