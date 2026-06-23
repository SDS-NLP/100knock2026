from sst2_common import load_sst2, make_features, TRAIN_PATH, DEV_PATH

# 61. 特徴ベクトル
# BoW（スペース区切りトークンの出現頻度）で各事例を特徴ベクトルに変換し、
# {text, label, feature} の辞書のリストにする。読み込み・特徴化は sst2_common に集約。

train_data = make_features(load_sst2(TRAIN_PATH))
dev_data = make_features(load_sst2(DEV_PATH))

# 学習データ先頭事例で変換を目視確認
print(train_data[0])


"""
{'text': 'hide new secretions from the parental units ', 'label': '0', 'feature': {'hide': 1, 'new': 1, 'secretions': 1, 'from': 1, 'the': 1, 'parental': 1, 'units': 1}}
"""
