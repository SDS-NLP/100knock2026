import csv
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# 63. 予測
# 学習したモデルで検証データ先頭の事例を予測し、付与済みの正解ラベルと一致するか確認する。

TRAIN_PATH = "SST-2/train.tsv"
DEV_PATH = "SST-2/dev.tsv"


def load_sst2(path):
    # SST-2 の tsv を読み込む（先頭行はヘッダ "sentence\tlabel"）
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        return [(sentence, label) for sentence, label in reader]


def make_features(data):
    # 各事例を BoW 頻度辞書つきの辞書にする（61と同じ）
    return [{"text": s, "label": l, "feature": dict(Counter(s.split()))} for s, l in data]


train_data = make_features(load_sst2(TRAIN_PATH))
dev_data = make_features(load_sst2(DEV_PATH))

# 62と同じ手順で学習（各スクリプトは独立なので再学習する）
vec = DictVectorizer()
X_train = vec.fit_transform([d["feature"] for d in train_data])
y_train = [d["label"] for d in train_data]
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 検証データ先頭事例を予測（学習時と同じ vec で transform するのが肝）
X_dev = vec.transform([d["feature"] for d in dev_data])
pred = clf.predict(X_dev[0])[0]
gold = dev_data[0]["label"]
print(f"text : {dev_data[0]['text']}")
print(f"予測 : {pred} / 正解 : {gold} / 一致 : {pred == gold}")


"""
text : it 's a charming and often affecting journey .
予測 : 1 / 正解 : 1 / 一致 : True
"""
