from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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
y_pred_train = model.predict(x_train)
y_pred_dev = model.predict(x_dev)

# print(classification_report(y_train, y_pred_train))
# print(classification_report(y_dev, y_pred_dev))

acc_train = accuracy_score(y_train, y_pred_train)
acc_dev = accuracy_score(y_dev, y_pred_dev)

prec_train = precision_score(y_train, y_pred_train, pos_label='1')
prec_dev = precision_score(y_dev, y_pred_dev, pos_label='1')

rec_train = recall_score(y_train, y_pred_train, pos_label='1')
rec_dev = recall_score(y_dev, y_pred_dev, pos_label='1')

f1_train = f1_score(y_train, y_pred_train, pos_label='1')
f1_dev = f1_score(y_dev, y_pred_dev, pos_label='1')

print("train, dev")
print(f"acc: {acc_train}, {acc_dev}")
print(f"prec: {prec_train}, {prec_dev}")
print(f"rec_train: {rec_train}, {rec_dev}")
print(f"f1: {f1_train}, {f1_dev}")

"""           precision    recall  f1-score   support

           0       0.94      0.93      0.93     29780
           1       0.94      0.95      0.95     37569

    accuracy                           0.94     67349
   macro avg       0.94      0.94      0.94     67349
weighted avg       0.94      0.94      0.94     67349

              precision    recall  f1-score   support

           0       0.82      0.79      0.80       428
           1       0.80      0.84      0.82       444

    accuracy                           0.81       872
   macro avg       0.81      0.81      0.81       872
weighted avg       0.81      0.81      0.81       872"""


"""train, dev
acc: 0.9419590491321326, 0.8119266055045872
prec: 0.9424522845575477, 0.8017241379310345
rec_train: 0.9542175729990151, 0.8378378378378378
f1: 0.9482984379753198, 0.8193832599118943"""