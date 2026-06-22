import pandas as pd
import numpy as np
from collections import Counter
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

text = "the worst movie I ‘ve ever seen"

df_train = pd.read_csv("SST-2/SST-2/train.tsv", sep='\t')
df_dev = pd.read_csv("SST-2/SST-2/dev.tsv", sep='\t')

def convert2dic(df):
    l = df['sentence']
    dic_list = []
    for i in range(len(l)):
        text = df.iloc[i]['sentence']
        d = dict()
        d['text'] = str(text)
        d['label'] = str(df.iloc[i]['label'])
        text_l = text.split()
        d['feature'] = dict(Counter(text_l))
        dic_list.append(d)
    return dic_list

train_list = convert2dic(df_train)
dev_list = convert2dic(df_dev)

logit_model = sklearn.linear_model.LogisticRegression(max_iter = 1000)



X_train_dict = [d['feature'] for d in train_list]
y_train = [int(d['label']) for d in train_list]

X_dev_dict = [d['feature'] for d in dev_list]
y_dev = [int(d['label']) for d in dev_list]

vec = DictVectorizer()
X_train = vec.fit_transform(X_train_dict)
X_dev = vec.transform(X_dev_dict)

logit_model.fit(X_train, y_train)


C_list = [0.01, 0.1, 1.0, 10.0, 100.0]  # 5点くらいで十分
train_accs, dev_accs = [], []

for C in C_list:
    model = sklearn.linear_model.LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, model.predict(X_train)))
    dev_accs.append(accuracy_score(y_dev, model.predict(X_dev)))
    print(f"C={C:>7}: train={train_accs[-1]:.4f}, dev={dev_accs[-1]:.4f}")

plt.figure(figsize=(7, 4))
plt.plot(C_list, train_accs, 'o-', label='Train')
plt.plot(C_list, dev_accs, 's-', label='Dev')
plt.xscale('log')  # Cは桁が変わるので対数軸
plt.xlabel('Regularization parameter C')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()