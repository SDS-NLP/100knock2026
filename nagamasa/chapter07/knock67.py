from sklearn.metrics import confusion_matrix

from sst2_common import DEV_PATH, TRAIN_PATH, load_sst2, make_features, train_model

# 67. 精度の計測
# 学習済みロジスティック回帰の 正解率/適合率/再現率/F1 を、学習データと検証データの両方で測る。


def compute_metrics(cm):
    # cm = confusion_matrix(y_true, y_pred, labels=["0","1"])  行=正解 / 列=予測
    # [0][0]=TN  [0][1]=FP / [1][0]=FN  [1][1]=TP
    ac = (cm[1][1] + cm[0][0]) / cm.sum()
    pre = cm[1][1] / (cm[0][1] + cm[1][1])
    re = cm[1][1] / (cm[1][0] + cm[1][1])
    f1 = 2 * pre * re / (pre + re)
    return ac, pre, re, f1


def main():
    # 学習済みモデルと train/dev を用意（train_model を再利用）
    vec, clf = train_model(make_features(load_sst2(TRAIN_PATH)))
    train_data = make_features(load_sst2(TRAIN_PATH))
    dev_data = make_features(load_sst2(DEV_PATH))

    # dev: 予測 → 混同行列
    X_dev = vec.transform([d["feature"] for d in dev_data])
    y_pred_dev = clf.predict(X_dev)
    y_true_dev = [d["label"] for d in dev_data]
    cm_dev = confusion_matrix(y_true_dev, y_pred_dev, labels=["0", "1"])

    # train: 予測 → 混同行列
    X_train = vec.transform([d["feature"] for d in train_data])
    y_pred_train = clf.predict(X_train)
    y_true_train = [d["label"] for d in train_data]
    cm_train = confusion_matrix(y_true_train, y_pred_train, labels=["0", "1"])

    # 4指標を train / dev 横並びで出力（trainだけ高い＝過学習が見える）
    names = ["accuracy", "precision", "recall", "f1"]
    train_scores = compute_metrics(cm_train)
    dev_scores = compute_metrics(cm_dev)

    print(f"{'':<12}{'train':>10}{'dev':>10}")
    for name, t, d in zip(names, train_scores, dev_scores):
        print(f"{name:<12}{t:>10.4f}{d:>10.4f}")


if __name__ == "__main__":
    main()


"""
                 train       dev
accuracy        0.9420    0.8119
precision       0.9425    0.8017
recall          0.9542    0.8378
f1              0.9483    0.8194
"""
