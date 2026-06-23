from sst2_common import load_sst2, make_features, train_model, TRAIN_PATH, DEV_PATH

# 63. 予測
# 学習したモデルで検証データ先頭の事例を予測し、付与済みの正解ラベルと一致するか確認する。

vec, clf = train_model(make_features(load_sst2(TRAIN_PATH)))
dev_data = make_features(load_sst2(DEV_PATH))

# 検証データは学習時と同じ vec で transform するのが肝
X_dev = vec.transform([d["feature"] for d in dev_data])
pred = clf.predict(X_dev[0])[0]
gold = dev_data[0]["label"]
print(f"text : {dev_data[0]['text']}")
print(f"予測 : {pred} / 正解 : {gold} / 一致 : {pred == gold}")


"""
text : it 's a charming and often affecting journey .
予測 : 1 / 正解 : 1 / 一致 : True
"""
