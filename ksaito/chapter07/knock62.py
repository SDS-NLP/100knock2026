from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sst2 import TRAIN_PATH
from knock61 import convert_examples


def train_model(train):
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(example['feature'] for example in train)
    y_train = [example['label'] for example in train]

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model, vectorizer


if __name__ == '__main__':
    train = convert_examples(TRAIN_PATH)
    model, vectorizer = train_model(train)
    print(f'features: {len(vectorizer.feature_names_)}')
    print(f'labels: {[str(label) for label in model.classes_]}')
