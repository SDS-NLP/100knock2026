from collections import Counter
import re

import MeCab
import unidic_lite
import matplotlib.pyplot as plt


def extract_words(text, tagger):
    """テキストから単語を取り出す"""
    words = []

    node = tagger.parseToNode(text)

    while node:
        surface = node.surface

        if surface:
            features = node.feature.split(",")
            pos = features[0]

            # 記号は除外
            if pos != "補助記号":
                words.append(surface)

        node = node.next

    return words


def main():
    input_path = "kokoro.txt"
    output_path = "word_frequency_loglog.png"

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 青空文庫のルビを削除
    text = re.sub(r"《.*?》", "", text)

    # 青空文庫の区切り記号を削除
    text = text.replace("｜", "")

    tagger = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")

    words = extract_words(text, tagger)

    counter = Counter(words)

    # 出現頻度の高い順に並べる
    frequencies = [freq for word, freq in counter.most_common()]
    ranks = range(1, len(frequencies) + 1)

    plt.figure(figsize=(8, 6))

    # 両対数グラフ
    plt.xscale("log")
    plt.yscale("log")

    plt.plot(ranks, frequencies, marker=".")

    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Word Frequency Ranking")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()