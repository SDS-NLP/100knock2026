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

predicted_label = model_logistic.predict(X)

first_sentence = dev_data[0]["text"]
first_label = dev_data[0]["label"]

print(
    f"文: {first_sentence}\n実際のラベル: {first_label}\n予測されたラベル: {predicted_label}\n予測は正しいか: {predicted_label == first_label}"
)

# ◎ uv run knock63.py
# 文: it 's a charming and often affecting journey .
# 実際のラベル: 1
# 予測されたラベル: [1]
# 予測は正しいか: [ True]
