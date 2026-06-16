from sst2 import DEV_PATH, TRAIN_PATH
from knock61 import convert_examples
from knock62 import train_model


def predict_probability(model, vectorizer, example):
    x = vectorizer.transform([example['feature']])
    return model.predict_proba(x)[0]


if __name__ == '__main__':
    train = convert_examples(TRAIN_PATH)
    dev = convert_examples(DEV_PATH)
    model, vectorizer = train_model(train)

    example = dev[0]
    probabilities = predict_probability(model, vectorizer, example)
    print(f'text: {example["text"]}')
    for label, probability in zip(model.classes_, probabilities):
        print(f'P(label={label} | text) = {probability:.4f}')
