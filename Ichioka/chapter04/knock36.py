from collections import Counter
import MeCab
import unidic_lite


def main():
    input_path = "kokoro.txt"

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    tagger = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")

    words = []

    node = tagger.parseToNode(text)

    while node:
        surface = node.surface

        if surface:
            features = node.feature.split(",")
            pos = features[0]

            if pos != "補助記号": # 記号や空白っぽいものは除外
                words.append(surface)

        node = node.next

    counter = Counter(words)

    for word, freq in counter.most_common(20):
        print(f"{word}\t{freq}")


if __name__ == "__main__":
    main()