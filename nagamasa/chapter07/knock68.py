from sst2_common import TRAIN_PATH, load_sst2, make_features, train_model

# 68. 特徴量の重みの確認
# 学習済みロジスティック回帰の、重みの高い特徴量トップ20と、低いトップ20を確認する。


def main():
    # 学習済みモデルを用意（train_model を再利用）。重みは clf.coef_ に入っている。
    vec, clf = train_model(make_features(load_sst2(TRAIN_PATH)))

    # 重み(coef_)と特徴名は列順で対応。2値分類なので coef_ は shape (1, n_features)。
    weight = clf.coef_[0]
    name = vec.get_feature_names_out()

    # 重みの昇順インデックス。末尾=最大(正)→上位、先頭=最小(負)→下位。
    order = weight.argsort()
    top20 = order[-20:][::-1]
    worst20 = order[:20]

    # coef_ は classes_[1] 基準。['0','1'] なら 重み高=ポジ(1) / 重み低=ネガ(0)。
    print("classes_:", clf.classes_)

    print("\n重みの高い特徴量トップ20")
    for n, w in zip(name[top20], weight[top20]):
        print(f"{n}\t{w:.4f}")

    print("\n重みの低い特徴量トップ20")
    for n, w in zip(name[worst20], weight[worst20]):
        print(f"{n}\t{w:.4f}")


if __name__ == "__main__":
    main()


"""
classes_: ['0' '1']

重みの高い特徴量トップ20
refreshing	3.3932
remarkable	3.3600
powerful	3.2024
hilarious	3.1376
beautiful	2.9787
wonderful	2.9538
prose	2.8445
appealing	2.8332
terrific	2.8174
treat	2.7850
enjoyable	2.7687
charmer	2.7405
vividly	2.6932
charming	2.6731
likable	2.6511
solid	2.6131
intriguing	2.5869
impressive	2.5802
half-bad	2.5576
fascinating	2.5539

重みの低い特徴量トップ20
lacking	-4.2836
worst	-4.0423
lacks	-4.0147
devoid	-3.6200
mess	-3.5397
failure	-3.5035
stupid	-3.3005
bore	-3.2129
flat	-3.2106
waste	-3.1383
loses	-3.1286
depressing	-3.1221
lack	-3.0221
none	-3.0119
squanders	-2.9805
hardly	-2.9739
poor	-2.9526
pointless	-2.9226
clichés	-2.9139
unfortunately	-2.9132
"""
