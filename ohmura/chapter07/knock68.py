import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('SST-2/train.tsv', sep='\t')

X_train_dict = [dict(Counter(str(row['sentence']).split())) for _, row in train_df.iterrows()]
y_train = [str(row['label']) for _, row in train_df.iterrows()]

vec = DictVectorizer()
X_train = vec.fit_transform(X_train_dict)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

feature_names = vec.get_feature_names_out()
weights = clf.coef_[0]

features_with_weights = list(zip(feature_names, weights))

sorted_features = sorted(features_with_weights, key=lambda x: x[1], reverse=True)

print("【重みの高い特徴量トップ20（ポジティブに寄与）】")
for i, (feature, weight) in enumerate(sorted_features[:20], 1):
    print(f"{i:2d}. {feature:<15} : {weight:.4f}")

print("\n【重みの低い特徴量トップ20（ネガティブに寄与）】")
for i, (feature, weight) in enumerate(sorted_features[-20:][::-1], 1):
    print(f"{i:2d}. {feature:<15} : {weight:.4f}")