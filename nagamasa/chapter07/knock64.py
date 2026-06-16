import csv
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# 64. 条件付き確率
# 学習したモデルで、検証データ先頭の事例を各ラベルに分類するときの条件付き確率を求める。

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

# 検証データ先頭事例の各ラベルの条件付き確率（predict_proba は clf.classes_ の順に返す）
X_dev = vec.transform([d["feature"] for d in dev_data])
proba = clf.predict_proba(X_dev[0])[0]
print(f"text : {dev_data[0]['text']}")
for cls, p in zip(clf.classes_, proba):
    label_name = "ポジ" if cls == "1" else "ネガ"
    print(f"P(label={cls} {label_name}) = {p:.4f}")


"""
text : it 's a charming and often affecting journey .
P(label=0 ネガ) = 0.0041
P(label=1 ポジ) = 0.9959
"""
