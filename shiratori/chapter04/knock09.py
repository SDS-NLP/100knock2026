import MeCab
import matplotlib.pyplot as plt
import math
from collections import Counter


def tokenize(text):
    mecab = MeCab.Tagger("-r /opt/homebrew/etc/mecabrc")
    node = mecab.parseToNode(text)

    words = []

    while node:
        if node.surface:
            features = node.feature.split(",")
            pos = features[0]
            subtype = features[1]

            if pos == "名詞":
                if subtype not in ["非自立", "代名詞", "数"]:
                    words.append(node.surface)

        node = node.next

    return words


def main():
    with open("chapter04/kokoro.txt", "r", encoding="utf-8") as f:
        text = f.read()

    words = tokenize(text)

    counter = Counter(words)

    freqs = sorted(counter.values(), reverse=True)

    ranks = list(range(1, len(freqs) + 1))

    log_ranks = [math.log10(r) for r in ranks]
    log_freqs = [math.log10(f) for f in freqs]

    plt.figure()
    plt.plot(log_ranks, log_freqs, marker=".")

    plt.xlabel("log(rank)")
    plt.ylabel("log(frequency)")
    plt.title("Zipf's Law (Kokoro)")

    plt.show()


if __name__ == "__main__":
    main()
