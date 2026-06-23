import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_df = pd.read_csv('SST-2/train.tsv', sep='\t')
dev_df = pd.read_csv('SST-2/dev.tsv', sep='\t')

X_train_dict = [dict(Counter(str(row['sentence']).split())) for _, row in train_df.iterrows()]
y_train = [str(row['label']) for _, row in train_df.iterrows()]

X_dev_dict = [dict(Counter(str(row['sentence']).split())) for _, row in dev_df.iterrows()]
y_dev = [str(row['label']) for _, row in dev_df.iterrows()]

vec = DictVectorizer()
X_train = vec.fit_transform(X_train_dict)
X_dev = vec.transform(X_dev_dict)

C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
accuracies = []

for c in C_values:
    clf = LogisticRegression(C=c, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    accuracies.append(acc)
    print(f"C={c:<7} : 正解率 {acc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(C_values, accuracies, marker='o', linestyle='-', color='b')
plt.xscale('log')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs Regularization Parameter')
plt.grid(True, which="both", ls="--")
plt.savefig('knock69.png')