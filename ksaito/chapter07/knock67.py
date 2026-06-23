from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from knock61 import convert_examples
from knock62 import train_model
from knock66 import predict_labels
from sst2 import DEV_PATH, TRAIN_PATH, download_sst2


def evaluate(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label='1'),
        'recall': recall_score(y_true, y_pred, pos_label='1'),
        'f1': f1_score(y_true, y_pred, pos_label='1'),
    }


def print_scores(name, scores):
    print(f'{name}:')
    for metric, score in scores.items():
        print(f'{metric}: {score:.4f}')


def main():
    download_sst2()
    train = convert_examples(TRAIN_PATH)
    dev = convert_examples(DEV_PATH)

    model, vectorizer = train_model(train)

    for name, dataset in [('train', train), ('dev', dev)]:
        y_true = [item['label'] for item in dataset]
        y_pred = predict_labels(model, vectorizer, dataset)
        scores = evaluate(y_true, y_pred)
        print_scores(name, scores)


if __name__ == '__main__':
    main()
