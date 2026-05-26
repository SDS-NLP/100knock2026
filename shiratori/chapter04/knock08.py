import MeCab
import re
import math
from collections import Counter, defaultdict


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
                words.append(node.surface)

        node = node.next

    return words


def split_sentences(text):
    return [s for s in re.split(r"。", text) if s.strip()]


def main():
    with open("chapter04/kokoro.txt", "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_sentences(text)

    docs = [tokenize(s) for s in sentences]

    N = len(docs)

    # TF
    tfs = []
    df = defaultdict(int)

    for doc in docs:
        tf = Counter(doc)
        tfs.append(tf)

        for word in set(doc):
            df[word] += 1

    # TF-IDF
    scores = defaultdict(float)

    for i, tf in enumerate(tfs):
        total = sum(tf.values())

        for word, count in tf.items():
            tf_val = count / total
            idf_val = math.log(N / (df[word] + 1))
            scores[word] += tf_val * idf_val

    # 上位20
    top20 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]

    print("word\tTF-IDF")
    for w, s in top20:
        tf = sum(tf[w] for tf in tfs)
        idf = math.log(N / (df[w] + 1))
        print(f"{w}\tTF:{tf}\tIDF:{idf:.4f}\tTF-IDF:{s:.4f}")


if __name__ == "__main__":
    main()
