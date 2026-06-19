from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from knock61 import objectify

train_data = objectify("SST-2/train.tsv")

model_logistic = LogisticRegression(max_iter=1000)

features = [item["feature"] for item in train_data]
y = [item["label"] for item in train_data]

vec = DictVectorizer(sparse=True)
X = vec.fit_transform(features)

model_logistic.fit(X, y)
predicted_y = model_logistic.predict(X)

dev_data = objectify("SST-2/dev.tsv")
dev_feature = [item["feature"] for item in dev_data]
dev_label = [item["label"] for item in dev_data]

X_dev = vec.transform(dev_feature)
predicted_label = model_logistic.predict(X_dev)

print(
    f"学習データ\n正解率: {accuracy_score(y, predicted_y)}\n適合率: {precision_score(y, predicted_y)}\n再現率: {recall_score(y, predicted_y)}\nF1値: {f1_score(y, predicted_y)}"
)

# result
# 学習データ
# 正解率: 0.9420184412537677
# 適合率: 0.9424815983175605
# 再現率: 0.9542974260693657
# F1値: 0.948352709333545

print(
    f"検証データ\n正解率: {accuracy_score(dev_label, predicted_label)}\n適合率: {precision_score(dev_label, predicted_label)}\n再現率: {recall_score(dev_label, predicted_label)}\nF1値: {f1_score(dev_label, predicted_label)}"
)

# result
# 検証データ
# 正解率: 0.8107798165137615
# 適合率: 0.8012958963282938
# 再現率: 0.8355855855855856
# F1値: 0.8180815876515987
