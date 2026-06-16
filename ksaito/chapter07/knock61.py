from collections import Counter

from sst2 import DEV_PATH, TRAIN_PATH, read_sst2


def extract_feature(text):
    return dict(Counter(text.split()))


def convert_examples(path):
    examples = []
    for text, label in read_sst2(path):
        examples.append({
            'text': text,
            'label': label,
            'feature': extract_feature(text),
        })

    return examples


if __name__ == '__main__':
    train = convert_examples(TRAIN_PATH)
    dev = convert_examples(DEV_PATH)

    print(f'train examples: {len(train)}')
    print(f'dev examples: {len(dev)}')
    print(train[0])
