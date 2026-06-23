from sklearn.metrics import confusion_matrix

from sst2_common import DEV_PATH, TRAIN_PATH, load_sst2, make_features, train_model

# 66. 混同行列の作成
# 学習済みロジスティック回帰の、検証データ(dev)における混同行列を求める。


def main():
    # 学習済みモデルと dev データを用意（train_model を再利用）
    vec, clf = train_model(make_features(load_sst2(TRAIN_PATH)))
    dev_data = make_features(load_sst2(DEV_PATH))

    # dev 全件を予測し、正解ラベルと突き合わせて混同行列を作る。
    # 行=正解・列=予測。labels で軸の順を 0(neg)→1(pos) に固定する。
    X_dev = vec.transform([d["feature"] for d in dev_data])
    y_pred = clf.predict(X_dev)
    y_true = [d["label"] for d in dev_data]
    cm = confusion_matrix(y_true, y_pred, labels=["0", "1"])

    # ラベル付きで見やすく表示する
    print("混同行列 (dev)  行=正解 / 列=予測")
    print("            pred=neg  pred=pos")
    print(f"true=neg    {cm[0][0]:>8}  {cm[0][1]:>8}")
    print(f"true=pos    {cm[1][0]:>8}  {cm[1][1]:>8}")


if __name__ == "__main__":
    main()


"""
混同行列 (dev)  行=正解 / 列=予測
            pred=neg  pred=pos
true=neg         336        92
true=pos          72       372
"""
