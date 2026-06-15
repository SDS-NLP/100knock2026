from datasets import load_dataset
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

dataset = load_dataset('glue', 'sst2')

def to_bow(text):
    return dict(Counter(text.lower().split()))

train_bow = [to_bow(ex['sentence']) for ex in dataset['train']]
train_y = [ex['label'] for ex in dataset['train']]
dev_bow = [to_bow(ex['sentence']) for ex in dataset['validation']]
dev_y = [ex['label'] for ex in dataset['validation']]

vec = DictVectorizer()
X_train = vec.fit_transform(train_bow)
X_dev = vec.transform(dev_bow)

C_values = np.logspace(-3, 3, 13)
train_accs, dev_accs = [], []

for C in C_values:
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(X_train, train_y)
    train_accs.append(accuracy_score(train_y, clf.predict(X_train)))
    dev_accs.append(accuracy_score(dev_y, clf.predict(X_dev)))
    print(f'C={C:.4f}  train={train_accs[-1]:.4f}  dev={dev_accs[-1]:.4f}')

plt.figure(figsize=(8, 5))
plt.semilogx(C_values, train_accs, label='train')
plt.semilogx(C_values, dev_accs, label='validation')
plt.xlabel('Regularization parameter C')
plt.ylabel('Accuracy')
plt.title('Logistic Regression: Accuracy vs Regularization')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('knock69_accuracy.png')
plt.show()
print('saved: knock69_accuracy.png')
