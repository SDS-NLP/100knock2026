from knock61 import convert_examples, extract_feature
from knock62 import train_model
from sst2 import TRAIN_PATH, download_sst2


TEXT = "the worst movie I 've ever seen"


def predict_text(model, vectorizer, text):
    feature = extract_feature(text)
    x = vectorizer.transform([feature])
    label = model.predict(x)[0]
    probabilities = model.predict_proba(x)[0]
    return label, probabilities


def main():
    download_sst2()
    train = convert_examples(TRAIN_PATH)
    model, vectorizer = train_model(train)
    label, probabilities = predict_text(model, vectorizer, TEXT)

    print(f'text: {TEXT}')
    print(f'predicted: {label}')
    for class_label, probability in zip(model.classes_, probabilities):
        print(f'P(label={class_label} | text) = {probability:.4f}')


if __name__ == '__main__':
    main()
