import matplotlib.pyplot as plt

from sst2_common import DEV_PATH, TRAIN_PATH, load_sst2, make_features, train_model

# 69. 正則化パラメータの変更
# 正則化係数 C を変えてロジスティック回帰を学習し、検証データ上の正解率をグラフにまとめる。
# C は正則化の逆数。C小=正則化強(未学習寄り) / C大=正則化弱(過学習寄り)。


def main():
    train_data = make_features(load_sst2(TRAIN_PATH))
    dev_data = make_features(load_sst2(DEV_PATH))

    # C を桁で振り(等比)、各 C で学習 → dev の正解率を記録する。
    C_list = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    accs = []
    for c in C_list:
        vec, clf = train_model(train_data, C=c)
        X_dev = vec.transform([d["feature"] for d in dev_data])
        y_true = [d["label"] for d in dev_data]
        acc = clf.score(X_dev, y_true)
        accs.append(acc)

    # C は桁で動くので x 軸は対数。横軸=C(C小ほど正則化強)。
    plt.plot(C_list, accs, marker="o")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("dev accuracy")
    plt.grid(True)
    plt.savefig("knock69.png")


if __name__ == "__main__":
    main()
