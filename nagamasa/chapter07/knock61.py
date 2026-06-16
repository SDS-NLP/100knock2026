import csv
from collections import Counter

# 61. 特徴ベクトル
# Bag of Words（スペース区切りトークンの出現頻度）で各事例を特徴ベクトルに変換し、
# 各事例を {text, label, feature} の辞書にまとめ、データを辞書のリストとして表現する。

TRAIN_PATH = "SST-2/train.tsv"
DEV_PATH = "SST-2/dev.tsv"


def load_sst2(path):
    # SST-2 の tsv を読み込む（先頭行はヘッダ "sentence\tlabel"）
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        return [(sentence, label) for sentence, label in reader]


train = load_sst2(TRAIN_PATH)
dev = load_sst2(DEV_PATH)


def make_features(data):
    # 各事例を {text, label, feature(BoW頻度辞書)} の辞書にして並べる
    result = []
    for sentence, label in data:
        feature = dict(Counter(sentence.split()))
        result.append({"text": sentence, "label": label, "feature": feature})
    return result


train_data = make_features(train)
dev_data = make_features(dev)

# 学習データ先頭事例で変換を目視確認
print(train_data[0])


"""
{'text': 'hide new secretions from the parental units ', 'label': '0', 'feature': {'hide': 1, 'new': 1, 'secretions': 1, 'from': 1, 'the': 1, 'parental': 1, 'units': 1}}
"""
