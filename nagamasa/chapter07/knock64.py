from sst2_common import load_sst2, make_features, train_model, TRAIN_PATH, DEV_PATH

# 64. 条件付き確率
# 学習したモデルで、検証データ先頭の事例を各ラベルに分類するときの条件付き確率を求める。

vec, clf = train_model(make_features(load_sst2(TRAIN_PATH)))
dev_data = make_features(load_sst2(DEV_PATH))

# predict_proba は clf.classes_ の順に確率を返すので zip でラベルと対応させる
X_dev = vec.transform([d["feature"] for d in dev_data])
proba = clf.predict_proba(X_dev[0])[0]
print(f"text : {dev_data[0]['text']}")
for cls, p in zip(clf.classes_, proba):
    label_name = "ポジ" if cls == "1" else "ネガ"
    print(f"P(label={cls} {label_name}) = {p:.4f}")


"""
text : it 's a charming and often affecting journey .
P(label=0 ネガ) = 0.0041
P(label=1 ポジ) = 0.9959
"""
