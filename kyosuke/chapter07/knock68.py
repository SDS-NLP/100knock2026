import joblib

model = joblib.load('lr_model.joblib')
vec = joblib.load('vectorizer.joblib')

feature_names = vec.get_feature_names_out()
weights = model.coef_[0]
features_with_weights = list(zip(feature_names, weights))
sorted_features = sorted(features_with_weights, key=lambda x: x[1])
print(f'重みの高い特徴量')
for i, (word, weight) in enumerate(sorted_features[:-21:-1], 1):
    print(f"{i:2d}位: {word:<15} (重み: {weight:.4f})")

print(f'重みの低い特徴量')
for i, (word, weight) in enumerate(sorted_features[:20], 1):
    print(f"{i:2d}位: {word:<15} (重み: {weight:.4f})")