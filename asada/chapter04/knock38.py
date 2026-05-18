import math
import re
from pathlib import Path

import MeCab

text = Path("kokoro.txt").read_text().splitlines()
nails = []
for i in range(len(text) - 1):
    if text[i] == text[i + 1]:
        nails.append(i + 1)

chapters = []
pattern = re.compile(r"^[一二三四五六七八九十百]{1,3}$")
for j in range(len(nails) - 1):
    chunk = "".join(text[nails[j] + 1 : nails[j + 1] - 1])
    if not pattern.match(chunk):
        chapters.append(chunk)
chapters.append("".join(text[nails[-1] + 1 :]))

chapters = [re.sub(r"《.*?》", "", re.sub(r"｜", "", ch)) for ch in chapters]
corpus = "".join(chapters)

wakati = MeCab.Tagger()
node = wakati.parseToNode(corpus)
tokens = []
while node:
    features = node.feature.split(",")
    if node.surface != "" and features[0] == "名詞":
        tokens.append((node.surface, features[7])) if len(
            features
        ) > 7 else tokens.append((node.surface, node.surface))
    node = node.next

tf = {}
for token in tokens:
    if token[1] not in tf:
        tf[token[1]] = 1
    else:
        tf[token[1]] += 1

idf = {
    token[1]: math.log(len(chapters) / sum(1 for ch in chapters if token[0] in ch))
    for token in set(tokens)
}


tf_idf = {}
for word in tf:
    score1 = tf[word]
    score2 = idf[word]
    score3 = tf[word] * idf[word]
    tf_idf[word] = (score1, score2, score3)

tf_idf = {k: v for k, v in sorted(tf_idf.items(), key=lambda x: x[1][2], reverse=True)}
print(
    "\n".join(
        f"第{rank + 1}位\t「{word}」\t TF: {tf_idf[word][0]}\t IDF: {tf_idf[word][1]}\t TF-IDF: {tf_idf[word][2]}"
        for rank, word in zip(range(20), tf_idf)
    )
)
