import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(file_path):
    """ファイルからデータを読み込み、特徴量辞書とラベルのリストを返す"""
    df = pd.read_csv(file_path, sep='\t')
    X_dicts = []
    y = []
    for _, row in df.iterrows():
        text = str(row['sentence'])
        X_dicts.append(dict(Counter(text.split())))
        y.append(int(row['label']))
    return X_dicts, y

X_train_dicts, y_train = load_data('SST-2/train.tsv')
X_dev_dicts, y_dev = load_data('SST-2/dev.tsv')

vec = DictVectorizer()
X_train = vec.fit_transform(X_train_dicts)
X_dev = vec.transform(X_dev_dicts)

C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
accuracies = []

for c in C_values:
    model = LogisticRegression(C=c, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    accuracies.append(acc)
    print(f"パラメータ C={c:<7} | 検証データ正解率: {acc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(C_values, accuracies, marker='o', linestyle='-', color='b')
plt.xscale('log')
plt.xlabel('Regularization Parameter C (Log Scale)')
plt.ylabel('Validation Accuracy')
plt.title('Effect of Regularization Parameter C on Validation Accuracy')
plt.grid(True)
plt.show()