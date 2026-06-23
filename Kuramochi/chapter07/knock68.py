"よくわからず作った→要解説"

import joblib

# 学習済みモデルとベクトライザーの読み込み
loaded_model      = joblib.load('sst2_lr_model.pkl')
loaded_vectorizer = joblib.load('sst2_vectorizer.pkl')

# 1. モデルの重み（係数）と特徴量（単語）の名前を取得
weights       = loaded_model.coef_[0]
feature_names = loaded_vectorizer.get_feature_names_out()

# 2. 特徴量名と重みをペアにしてリスト化
feature_weights = list(zip(feature_names, weights))

# 3. 重みの値で降順（大きい順）にソート
sorted_features = sorted(feature_weights, key=lambda x: x[1], reverse=True)

# 4. 重みの高い特徴量トップ20（ポジティブに寄与）
top_20_positive = sorted_features[:20]
top_20_negative = sorted_features[-20:][::-1]

# 6. 結果の表示
print("【重みの高い特徴量トップ20（ポジティブ）】")
for rank, (feature, weight) in enumerate(top_20_positive, 1):
    print(f"{rank:2d}位: {feature:<15} ({weight:.4f})")

print("\n【重みの低い特徴量トップ20（ネガティブ）】")
for rank, (feature, weight) in enumerate(top_20_negative, 1):
    print(f"{rank:2d}位: {feature:<15} ({weight:.4f})")