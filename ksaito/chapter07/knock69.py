import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib')

import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from knock61 import convert_examples
from knock66 import predict_labels
from sst2 import DEV_PATH, TRAIN_PATH, download_sst2


C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
OUTPUT_PATH = 'regularization_accuracy.png'


def train_model_with_regularization(train, c):
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(item['feature'] for item in train)
    y_train = [item['label'] for item in train]

    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(x_train, y_train)
    return model, vectorizer


def calculate_accuracy(model, vectorizer, dataset):
    y_true = [item['label'] for item in dataset]
    y_pred = predict_labels(model, vectorizer, dataset)
    return accuracy_score(y_true, y_pred)


def plot_accuracy(c_values, accuracies):
    plt.plot(c_values, accuracies, marker='o')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.savefig(OUTPUT_PATH)


def main():
    download_sst2()
    train = convert_examples(TRAIN_PATH)
    dev = convert_examples(DEV_PATH)

    accuracies = []
    for c in C_VALUES:
        model, vectorizer = train_model_with_regularization(train, c)
        accuracy = calculate_accuracy(model, vectorizer, dev)
        accuracies.append(accuracy)
        print(f'C={c}: accuracy={accuracy:.4f}')

    plot_accuracy(C_VALUES, accuracies)
    print(f'saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
