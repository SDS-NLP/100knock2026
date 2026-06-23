from collections import Counter

from sst2_common import load_sst2, TRAIN_PATH, DEV_PATH

# 60. データの入手・整形
# SST-2 の train.tsv / dev.tsv について、ポジティブ(1)・ネガティブ(0)の事例数を数える。
# 読み込みは sst2_common.load_sst2 に集約。

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
