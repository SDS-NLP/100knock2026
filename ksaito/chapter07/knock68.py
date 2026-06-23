from knock61 import convert_examples
from knock62 import train_model
from sst2 import TRAIN_PATH, download_sst2


def get_feature_weights(model, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    return list(zip(feature_names, weights))


def print_top_features(title, feature_weights):
    print(title)
    for feature, weight in feature_weights:
        print(f'{feature}\t{weight:.4f}')


def main():
    download_sst2()
    train = convert_examples(TRAIN_PATH)
    model, vectorizer = train_model(train)

    feature_weights = get_feature_weights(model, vectorizer)
    sorted_weights = sorted(feature_weights, key=lambda x: x[1])

    print_top_features('lowest weights:', sorted_weights[:20])
    print()
    print_top_features('highest weights:', sorted_weights[-20:][::-1])


if __name__ == '__main__':
    main()
