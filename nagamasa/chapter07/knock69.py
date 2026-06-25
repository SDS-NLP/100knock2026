import matplotlib.pyplot as plt
from sst2_common import DEV_PATH, TRAIN_PATH, load_sst2, make_features, train_model

# 69. 正則化パラメータの変更
# 正則化係数を変えてロジスティック回帰を学習し、検証データ上の正解率をグラフにまとめる。


def main():
    # データを用意（特徴化まで）。
    # 注意: 学習は C を変えて繰り返すので、C を受けない sst2_common.train_model はそのままでは使えない。
    train_data = make_features(load_sst2(TRAIN_PATH))
    dev_data = make_features(load_sst2(DEV_PATH))

    # ===== ここから核：あなたが書く =====
    # 0. ★設計判断: C を変えて学習する構造をどう作るか。選択肢（推薦はしない）:
    #      - sst2_common.train_model に C 引数を足す（既定1.0なら 62-68 は無改変で動く）
    #      - knock69 内に DictVectorizer + LogisticRegression(C=c) を直書き
    #      - ベクトル化はループ外で1回だけ、ループ内は LogisticRegression(C=c) の学習だけ（Cと無関係なので無駄が出ない）
    #    ※ sklearn の C は「正則化の逆数」。C大=正則化弱(過学習寄り) / C小=正則化強(未学習寄り)。

    # 1. 試す C の集合を決める。桁で振るのが定番（等比。例 1e-3 〜 1e3）。
    C_list = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    # 2. 各 C について: 学習 → dev を予測 → 正解率を記録（C と accuracy のペアを貯める）。
    #    正解率は accuracy_score(y_true, y_pred) でも clf.score(X_dev, y_true) でも可。
    accs = []
    for c in C_list:
        vec, clf = train_model(train_data, C=c)
        X_dev = vec.transform([d["feature"] for d in dev_data])
        y_true = [d["label"] for d in dev_data]
        acc = clf.score(X_dev, y_true)
        accs.append(acc)

    # 3. グラフ化。C は桁で動くので x 軸は対数スケール。横軸=C か 1/C かは自由、軸ラベルで向きを明示。

    # ===== 核ここまで =====

    # --- 雑用: プロット（横軸 C は桁で動くので対数。山型＝小Cで未学習・大Cで過学習）---
    plt.plot(C_list, accs, marker="o")
    plt.xscale("log")
    plt.xlabel("C")            # 横軸=C（C小ほど正則化強）。1/C にするなら値も反転させる
    plt.ylabel("dev accuracy")
    plt.grid(True)
    plt.savefig("knock69.png")


if __name__ == "__main__":
    main()
