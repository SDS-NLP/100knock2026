"""71. データセットの読み込み

GLUE で配布されている SST-2 の train.tsv / dev.tsv を読み込み、
テキストと極性ラベルを取得し、全テキストをトークンID列に変換する。
- 単語埋め込みの語彙に含まれない単語 (OOV) は無視し、トークン列に含めない。
- 全トークンが OOV で空のトークン列になった事例は削除する。

各事例は次のような辞書で表現する:
    {'text': 'contains no wit , only labored gags',
     'label': tensor([0.]),
     'input_ids': tensor([ 3475, 87, 15888, 90, 27695, 42637])}
"""

import csv

import torch

from knock70 import load_embeddings

DATA_DIR = "./SST-2"


def text_to_ids(text, word_to_id):
    """空白区切りのテキストをトークンID列に変換する (OOV は無視)。"""
    return [word_to_id[token] for token in text.split() if token in word_to_id]


def load_dataset(filepath, word_to_id):
    """SST-2 の tsv を読み込み、辞書のリストを返す。空のトークン列の事例は除外。"""
    dataset = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["sentence"].strip()
            input_ids = text_to_ids(text, word_to_id)
            if len(input_ids) == 0:
                continue  # 全トークンが語彙外の事例は削除
            dataset.append({
                "text": text,
                "label": torch.tensor([float(row["label"])]),
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
            })
    return dataset


if __name__ == "__main__":
    _, word_to_id, _ = load_embeddings()

    train = load_dataset(f"{DATA_DIR}/train.tsv", word_to_id)
    dev = load_dataset(f"{DATA_DIR}/dev.tsv", word_to_id)

    print(f"train size: {len(train)}")
    print(f"dev size:   {len(dev)}")
    print("\n-- first 2 train examples --")
    for ex in train[:2]:
        print(ex)
