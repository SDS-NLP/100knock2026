import MeCab
from collections import Counter
import re


def tokenize(text):
    mecab = MeCab.Tagger("-r /opt/homebrew/etc/mecabrc")
    node = mecab.parseToNode(text)

    words = []

    while node:
        features = node.feature.split(",")
        pos = features[0]

        if pos in ["名詞", "動詞", "形容詞"]:
            surface = node.surface
            if surface:
                words.append(surface)

        node = node.next

    return words


def main():
    with open("chapter04/kokoro.txt", "r", encoding="utf-8") as f:
        text = f.read()

    words = tokenize(text)

    counter = Counter(words)

    for word, freq in counter.most_common(20):
        print(word, freq)


if __name__ == "__main__":
    main()
