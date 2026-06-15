from sst2 import DEV_PATH, TRAIN_PATH
from knock61 import convert_examples
from knock62 import train_model


def predict_label(model, vectorizer, example):
    x = vectorizer.transform([example['feature']])
    return model.predict(x)[0]


if __name__ == '__main__':
    train = convert_examples(TRAIN_PATH)
    dev = convert_examples(DEV_PATH)
    model, vectorizer = train_model(train)

    example = dev[0]
    predicted = predict_label(model, vectorizer, example)
    print(f'text: {example["text"]}')
    print(f'gold: {example["label"]}')
    print(f'predicted: {predicted}')
    print(f'match: {predicted == example["label"]}')
