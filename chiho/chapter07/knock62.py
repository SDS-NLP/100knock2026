# 62. 学習
# 61で作成した BoW 特徴を使って、ロジスティック回帰モデルを学習する

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from knock61 import DATA_DIR, load_dataset

# 学習済みモデル，ベクトライザの保存先を指定
MODEL_PATH = DATA_DIR / "logistic_regression_sst2.pkl"
VECTORIZER_PATH = DATA_DIR / "sst2_vectorizer.pkl"


def vectorize_data(train_data, dev_data):
    vectorizer = DictVectorizer(sparse=False) # そこまで大規模ではないから密行列で保持
    # train: fit_transform でどの特徴がどの列か？＋行列変換
    train_x = vectorizer.fit_transform([d["feature"] for d in train_data])
    # dev: train での列に対応するよう，transformのみ
    dev_x = vectorizer.transform([d["feature"] for d in dev_data])
    return train_x, dev_x, vectorizer


def train_model(train_x, train_y):
    # 中規模までのデータに向いている重み最適化アルゴリズム
    model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
    model.fit(train_x, train_y)
    return model


def main() -> None:
    train_data = load_dataset(DATA_DIR / "train.tsv")
    dev_data = load_dataset(DATA_DIR / "dev.tsv")

    train_x, dev_x, vectorizer = vectorize_data(train_data, dev_data)

    train_y = [int(d["label"]) for d in train_data]
    dev_y = [int(d["label"]) for d in dev_data]

    model = train_model(train_x, train_y)

    accuracy = accuracy_score(dev_y, model.predict(dev_x))
    print(f"dev accuracy: {accuracy}")

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)


if __name__ == "__main__":
    main()
