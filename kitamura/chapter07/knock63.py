from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

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

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_dev)

correct = 0
mistake = 0
for i in range(len(y_dev)):
    if y_pred[i] == y_dev[i]:
        correct += 1

    else:
        mistake += 1

print(f"correct : {correct}, mistake : {mistake}")
print(f"accuracy : {correct/len(y_dev)}")

#correct : 708, mistake : 164
#accuracy : 0.8119266055045872