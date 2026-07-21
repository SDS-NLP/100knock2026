import csv
from transformers import AutoTokenizer

train_path = "/Users/caitlyn/Downloads/SST-2/train.tsv"
dev_path = "/Users/caitlyn/Downloads/SST-2/dev.tsv"
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def load_dataset(path):
    dataset = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            text = row["sentence"]
            label = int(row["label"])
            tokens = tokenizer.tokenize(text)

            dataset.append({
                "text": text,
                "label": label,
                "tokens": tokens,
            })

    return dataset

train_dataset = load_dataset(train_path)
dev_dataset = load_dataset(dev_path)

print(train_dataset[0])
