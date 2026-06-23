from sst2_common import load_sst2, make_features, train_model, TRAIN_PATH

# 62. 学習
# BoW特徴を使ってロジスティック回帰モデルを学習する。
# 読み込み・特徴化・学習は sst2_common に集約。

train_data = make_features(load_sst2(TRAIN_PATH))
vec, clf = train_model(train_data)

# 学習データ上の語彙数と正解率を確認（X_train は学習済み vec で transform して再現）
X_train = vec.transform([d["feature"] for d in train_data])
y_train = [d["label"] for d in train_data]
print(f"学習完了。語彙数(特徴量数)={len(vec.get_feature_names_out())}")
print(f"学習データ正解率: {clf.score(X_train, y_train):.4f}")


"""
学習完了。語彙数(特徴量数)=14816
学習データ正解率: 0.9420
"""
