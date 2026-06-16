from collections import Counter

from sst2 import DEV_PATH, TRAIN_PATH, read_sst2


def count_labels(path):
    examples = read_sst2(path)
    return Counter(label for _, label in examples)


if __name__ == '__main__':
    for name, path in [('train', TRAIN_PATH), ('dev', DEV_PATH)]:
        counts = count_labels(path)
        print(f'{name}: negative(0)={counts["0"]}, positive(1)={counts["1"]}')
