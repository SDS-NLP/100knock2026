import csv
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# 62. 学習
# BoW 特徴ベクトルを使ってロジスティック回帰モデルを学習する。

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

# BoW辞書のリストを疎行列に変換（DictVectorizer が語彙→列を対応づける）
vec = DictVectorizer()
X_train = vec.fit_transform([d["feature"] for d in train_data])
y_train = [d["label"] for d in train_data]

# ロジスティック回帰を学習（BoWは高次元疎なので反復多めで収束させる）
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print(f"学習完了。語彙数(特徴量数)={len(vec.get_feature_names_out())}")
print(f"学習データ正解率: {clf.score(X_train, y_train):.4f}")


"""
学習完了。語彙数(特徴量数)=14816
学習データ正解率: 0.9420
"""
