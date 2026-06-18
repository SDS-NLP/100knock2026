from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from knock61 import objectify

train_data = objectify("SST-2/train.tsv")

model_logistic = LogisticRegression(max_iter=1000)

features = [item["feature"] for item in train_data]
y = [item["label"] for item in train_data]

vec = DictVectorizer(sparse=True)
X = vec.fit_transform(features)

model_logistic.fit(X, y)

dev_data = objectify("SST-2/dev.tsv")
dev_feature = [item["feature"] for item in dev_data]
dev_label = [item["label"] for item in dev_data]

X = vec.transform(dev_feature)
predicted_label = model_logistic.predict(X)

cm = confusion_matrix(y_true=dev_label, y_pred=predicted_label)
print(cm)

# result
# [[336  92]
#  [ 73 371]]
