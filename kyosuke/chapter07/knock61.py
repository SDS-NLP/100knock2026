import pandas as pd
from collections import Counter

def make_list(file_path):
    dataset = []
    df = pd.read_csv(file_path, sep='\t')
    for _, row in df.iterrows():
        text = row["sentence"]
        label = row["label"]
        feature = dict(Counter(text.split()))
        instance = {
            "text": text,
            "label": label,
            "feature": feature
        }
        dataset.append(instance)
    return dataset

print(make_list("SST-2/train.tsv")[0])