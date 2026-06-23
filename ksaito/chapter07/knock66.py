from sklearn.metrics import confusion_matrix

from knock61 import convert_examples
from knock62 import train_model
from sst2 import DEV_PATH, TRAIN_PATH, download_sst2


def predict_labels(model, vectorizer, dataset):
    x = vectorizer.transform(item['feature'] for item in dataset)
    return model.predict(x)


def main():
    download_sst2()
    train = convert_examples(TRAIN_PATH)
    dev = convert_examples(DEV_PATH)

    model, vectorizer = train_model(train)
    y_true = [example['label'] for example in dev]
    y_pred = predict_labels(model, vectorizer, dev)

    print(confusion_matrix(y_true, y_pred, labels=['0', '1']))


if __name__ == '__main__':
    main()
