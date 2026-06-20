import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    X_dicts = []
    y = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)

        for row in reader:
            if len(row) != 2:
                continue
            text, label = row
            X_dicts.append(dict(Counter(text.split(' '))))
            y.append(label)

    return X_dicts, y

def main():
    train_file = './chapter07/SST-2/train.tsv'
    dev_file = './chapter07/SST-2/dev.tsv'

    train_X_dicts, y_train = load_data(train_file)
    dev_X_dicts, y_dev = load_data(dev_file)

    vec = DictVectorizer()
    X_train = vec.fit_transform(train_X_dicts)
    X_dev = vec.transform(dev_X_dicts)

    C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    train_accuracies = []
    dev_accuracies = []

    for C in C_values:
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_dev_pred = clf.predict(X_dev)

        train_acc = accuracy_score(y_train, y_train_pred)
        dev_acc = accuracy_score(y_dev, y_dev_pred)

        train_accuracies.append(train_acc)
        dev_accuracies.append(dev_acc)
        
        print(f"C = {C:<6} | テストデータ正解率: {train_acc:.4f} | 検証データ正解率: {dev_acc:.4f}")

    plt.figure(figsize=(10, 6))

    plt.plot(C_values, train_accuracies, marker='o', label='Train Accuracy', color='blue', linestyle='--')
    plt.plot(C_values, dev_accuracies, marker='s', label='Validation Accuracy', color='red')
    
    plt.xscale('log')

    plt.title("Effect of Regularization Parameter C on Accuracy")
    plt.xlabel("Regularization Parameter: C (log scale)")
    plt.ylabel("Accuracy")
    plt.grid(True, which="both", ls="-")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()