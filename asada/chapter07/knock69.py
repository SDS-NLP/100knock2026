import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from knock61 import objectify

train_data = objectify("SST-2/train.tsv")

features = [item["feature"] for item in train_data]
vec = DictVectorizer(sparse=True)
X = vec.fit_transform(features)
y = [item["label"] for item in train_data]

dev_data = objectify("SST-2/dev.tsv")
dev_feature = [item["feature"] for item in dev_data]
dev_label = [item["label"] for item in dev_data]
X_dev = vec.transform(dev_feature)

Cs = np.arange(0.1, 5.1, 0.1)
mls = [LogisticRegression(C=C, max_iter=1000).fit(X, y) for C in Cs]
dev_results = [accuracy_score(dev_label, ml.predict(X_dev)) for ml in mls]

plt.plot(Cs, dev_results)
plt.xlabel("正則化パラメータ", fontname="Noto Sans CJK JP")
plt.ylabel("正解率", fontname="Noto Sans CJK JP")
plt.savefig("result.png")
