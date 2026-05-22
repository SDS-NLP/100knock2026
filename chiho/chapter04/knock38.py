from collections import Counter
from math import log
from pathlib import Path
import re

import MeCab


def clean_text(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = re.sub(r"《[^》]+》", "", text)
    text = re.sub(r"［＃.*?］", "", text)
    text = re.sub(r"^.+?｜", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[一二三四五六七八九十百]+$", "", text, flags=re.MULTILINE)
    return text


def split_documents(text: str) -> list[str]:
    text = text.lstrip("\ufeff").replace("\r\n", "\n")
    parts = re.split(r"\n\s*[一二三四五六七八九十百]+\s*\n", text)
    return [clean_text(part).strip() for part in parts if clean_text(part).strip()]


def extract_independent_nouns(text: str, mecab: MeCab.Tagger) -> list[str]:
    nouns = []
    node = mecab.parseToNode(text)

    while node:
        if node.surface:
            features = node.feature.split(",")
            if features[0] == "名詞" and features[1] != "非自立":
                nouns.append(node.surface)
        node = node.next

    return nouns


file_path = Path(__file__).with_name("kokoro.txt")
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

documents = split_documents(raw_text)
mecab = MeCab.Tagger(r"-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/ipadic")

tf_counter = Counter()
df_counter = Counter()

for document in documents:
    nouns = extract_independent_nouns(document, mecab)
    tf_counter.update(nouns)
    df_counter.update(set(nouns))

total_terms = sum(tf_counter.values())
document_count = len(documents)

scores = []
for word, count in tf_counter.items():
    tf = count / total_terms
    idf = log(document_count / df_counter[word])
    tf_idf = tf * idf
    scores.append((word, tf, idf, tf_idf))

scores.sort(key=lambda item: item[3], reverse=True)

print("word\tTF\tIDF\tTF-IDF")
for word, tf, idf, tf_idf in scores[:20]:
    print(f"{word}\t{tf:.6f}\t{idf:.6f}\t{tf_idf:.6f}")
