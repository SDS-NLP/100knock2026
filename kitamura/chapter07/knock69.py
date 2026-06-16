from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train = []
with open("./SST-2/train.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        train.append(line.strip().split("\t"))

dev = []
with open("./SST-2/dev.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        dev.append(line.strip().split("\t"))


def create_dict_list(data):
    features = []
    labels = []
    for i in range(len(data)):
        tokens = data[i][0].split()
        feature_dict = dict(Counter(tokens))
        features.append(feature_dict)
        labels.append(data[i][1])
    return features, labels  


x_train_dict, y_train = create_dict_list(train)
x_dev_dict, y_dev = create_dict_list(dev)

vectorizer = DictVectorizer(sparse=True)
x_train = vectorizer.fit_transform(x_train_dict)
x_dev = vectorizer.transform(x_dev_dict)

c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
acc_list = []
for c in c_values:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_dev)
    acc = accuracy_score(y_dev, y_pred)
    acc_list.append(acc)

plt.plot(c_values, acc_list)
plt.xscale("log")
plt.show()