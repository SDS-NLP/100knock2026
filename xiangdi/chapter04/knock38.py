import re
import MeCab
from collections import Counter
from math import log

with open("/Users/caitlyn/Downloads/kokoro.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = re.sub(r"《.*?》", "", text)
text = re.sub(r"［＃.*?］", "", text)
text = text.replace("｜", "")
text = text.replace("\r", "")


documents = re.split(r"[。！？\n]", text)
documents = [doc.strip() for doc in documents if doc.strip()]

tagger = MeCab.Tagger()

doc_nouns = []
all_nouns = []

for doc in documents:
    node = tagger.parseToNode(doc)

    nouns = []

    while node:
        features = node.feature.split(",")

        if features[0] == "名詞":
            word = node.surface
            nouns.append(word)
            all_nouns.append(word)

        node = node.next
    doc_nouns.append(nouns)

    tf_counter = Counter(all_nouns)

total_words = sum(tf_counter.values())

tf = {
    word: count / total_words
    for word, count in tf_counter.items()
}

N = len(documents)

idf = {}

for word in tf_counter:
    df = sum(1 for doc in doc_nouns if word in doc)

    idf[word] = log(N / df) + 1

tfidf = {}

for word in tf_counter:
    tfidf[word] = tf[word] * idf[word]

top20 = sorted(
    tfidf.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]

print(f"{'単語':<15} {'TF':<10} {'IDF':<10} {'TF-IDF'}")

for word, score in top20:
    print(
        f"{word:<15} "
        f"{tf[word]:<10.6f} "
        f"{idf[word]:<10.6f} "
        f"{score:.6f}"
    )