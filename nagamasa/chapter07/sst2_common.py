import csv
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# 第7章 共通処理: SST-2 の読み込み・BoW特徴化・ロジスティック回帰の学習。
# knock60-64 で重複していた load_sst2 / make_features / train_model をここに集約する。
# 挙動は各knockの元コードと同一（csv / Counter / 疎行列のまま / lbfgs / ラベルは文字列 "0"/"1"）。

TRAIN_PATH = "SST-2/train.tsv"
DEV_PATH = "SST-2/dev.tsv"


def load_sst2(path):
    # SST-2 の tsv を読み込む（先頭行はヘッダ "sentence\tlabel"）。
    # 返り値: [(sentence, label), ...]  label は文字列 "0"/"1"。
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        return [(sentence, label) for sentence, label in reader]


def make_features(data):
    # 各事例を {text, label, feature(BoW頻度辞書)} の辞書にして並べる。
    # feature = dict(Counter(空白区切りトークン))。
    return [
        {"text": s, "label": l, "feature": dict(Counter(s.split()))}
        for s, l in data
    ]


def train_model(train_data, C=1.0):
    # BoW辞書のリストを疎行列化し、ロジスティック回帰を学習する。
    # C = 正則化の逆数。既定 1.0 は sklearn 既定と同じ → knock62-68 は呼び出し・結果とも無変化。
    # 返り値: (vectorizer, clf)。検証データは同じ vectorizer で transform すること。
    vec = DictVectorizer()
    X_train = vec.fit_transform([d["feature"] for d in train_data])
    y_train = [d["label"] for d in train_data]
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(X_train, y_train)
    return vec, clf
