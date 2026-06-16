import csv
from collections import Counter

# 60. データの入手・整形
# SST-2 の train.tsv / dev.tsv について、ポジティブ(1)・ネガティブ(0)の事例数を数える。

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

# ラベルごと（0/1）に事例数を数える
for name, data in [("train", train), ("dev", dev)]:
    counts = Counter(label for _, label in data)
    print(f"{name}: positive(1)={counts['1']}, negative(0)={counts['0']}, total={len(data)}")


"""
train: positive(1)=37569, negative(0)=29780, total=67349
dev: positive(1)=444, negative(0)=428, total=872
"""
