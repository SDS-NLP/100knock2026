from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from knock61 import objectify

train_data = objectify("SST-2/train.tsv")

model_logistic = LogisticRegression(max_iter=1000)

features = [item["feature"] for item in train_data]
y = [item["label"] for item in train_data]

vec = DictVectorizer(sparse=True)
X = vec.fit_transform(features)

model_logistic.fit(X, y)

dev_data = objectify("SST-2/dev.tsv")
dev_feature = dev_data[0]["feature"]

X = vec.transform([dev_feature])

predicted_prob = model_logistic.predict_proba(X)

print(f"条件付き確率: {predicted_prob}")
