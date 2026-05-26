from collections import Counter, defaultdict
import math
import re

import MeCab
import unidic_lite


def extract_nouns(text, tagger):
    """テキストから名詞だけを取り出す"""
    nouns = []

    node = tagger.parseToNode(text)

    while node:
        surface = node.surface

        if surface:
            features = node.feature.split(",")
            pos = features[0]

            if pos == "名詞":
                nouns.append(surface)

        node = node.next

    return nouns


def main():
    input_path = "kokoro.txt"

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ルビを削除：私《わたくし》 → 私
    text = re.sub(r"《.*?》", "", text)

    # 青空文庫によくある記号を軽く削除
    text = text.replace("｜", "")

    # 文単位で分割：各文を1文書とみなす
    documents = re.split(r"。|\n", text)
    documents = [doc.strip() for doc in documents if doc.strip()]

    tagger = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")

    # 各文書の名詞リスト
    noun_docs = [extract_nouns(doc, tagger) for doc in documents]
    noun_docs = [nouns for nouns in noun_docs if nouns]

    # 文書数
    N = len(noun_docs)

    # 全体での名詞出現回数
    total_noun_count = 0
    term_count = Counter()

    # 各名詞が何文書に出現したか
    document_frequency = defaultdict(int)

    for nouns in noun_docs:
        term_count.update(nouns)
        total_noun_count += len(nouns)

        for noun in set(nouns):
            document_frequency[noun] += 1

    results = []

    for noun, count in term_count.items():
        # TF: 全名詞中でその名詞が占める割合
        tf = count / total_noun_count

        # IDF: 多くの文書に出る語ほど小さくなる
        df = document_frequency[noun]
        idf = math.log(N / df) + 1

        tfidf = tf * idf

        results.append((noun, tf, idf, tfidf, count, df))

    # TF-IDFスコアで降順ソート
    results.sort(key=lambda x: x[3], reverse=True)

    print("名詞\tTF\tIDF\tTF-IDF\t出現回数\t文書頻度")
    print("-" * 60)

    for noun, tf, idf, tfidf, count, df in results[:20]:
        print(f"{noun}\t{tf:.6f}\t{idf:.6f}\t{tfidf:.6f}\t{count}\t{df}")


if __name__ == "__main__":
    main()