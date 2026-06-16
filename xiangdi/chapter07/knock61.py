import pandas as pd
import json
from collections import Counter

train_path = "/Users/caitlyn/Downloads/SST-2/train.tsv"
dev_path = "/Users/caitlyn/Downloads/SST-2/dev.tsv"

train_df = pd.read_csv(train_path, sep="\t")
dev_df = pd.read_csv(dev_path, sep="\t")

def text_to_feature(text):
    tokens = text.split()
    feature = dict(Counter(tokens))
    return feature

def load_sst2_as_dict_list(path):

    df = pd.read_csv(path, sep="\t")

    data = []

    for _, row in df.iterrows():
        text = row["sentence"]
        label = str(row["label"])

        example = {
            "text": text,
            "label": label,
            "feature": text_to_feature(text)
        }

        data.append(example)

    return data

train_data = load_sst2_as_dict_list(train_path)
dev_data = load_sst2_as_dict_list(dev_path)

print(train_data[0])

train_output_path = "/Users/caitlyn/Downloads/SST-2/train.json"
dev_output_path = "/Users/caitlyn/Downloads/SST-2/dev.json"

with open(train_output_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False)

with open(dev_output_path, "w", encoding="utf-8") as f:
    json.dump(dev_data, f, ensure_ascii=False)