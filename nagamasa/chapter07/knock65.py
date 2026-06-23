import argparse
from collections import Counter

from sst2_common import TRAIN_PATH, load_sst2, make_features, train_model

# 65. テキストのポジネガの予測
# 与えられた任意のテキストを、学習済みロジスティック回帰でポジ(1)/ネガ(0)に分類する。

EXAMPLE_TEXT = "the worst movie I 've ever seen"


def parse_args():
    # 入力テキストの受け口（省略時は問題の例文）
    p = argparse.ArgumentParser(description="入力テキストのポジ/ネガを予測する")
    p.add_argument("text", nargs="?", default=EXAMPLE_TEXT, help="予測対象のテキスト")
    return p.parse_args()


def main():
    args = parse_args()

    # 学習済みモデルを用意（train_model を再利用）
    vec, clf = train_model(make_features(load_sst2(TRAIN_PATH)))

    # 入力文を学習時と同じBoW（dict(Counter(split))）にし、学習済み vec で transform して予測する。
    # transform は「辞書のリスト」を取るので、1件でも [feature] で包む。
    feature = dict(Counter(args.text.split()))
    X = vec.transform([feature])
    pred = clf.predict(X)[0]
    print(pred)


if __name__ == "__main__":
    main()


"""
0
"""
