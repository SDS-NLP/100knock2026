import matplotlib.pyplot as plt
from collections import Counter

from morph_utils import parse_to_morphs


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_words(text):
    return [
        morph["surface"]
        for morph in parse_to_morphs(text)
        if morph["pos"] != "記号"
    ]

if __name__ == "__main__":
    path = "data/kokoro.txt"
    text = read_text(path)

    words = extract_words(text)
    word_frequency = Counter(words)

    frequencies = [
        count
        for _, count in word_frequency.most_common()
    ]

    ranks = range(1, len(frequencies) + 1)

    plt.plot(ranks, frequencies)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf's Law in Kokoro")
    plt.show()
