import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
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
    if node.surface != "":
        tokens.append(features[7]) if len(features) > 7 else tokens.append(node.surface)
    node = node.next

freq = {}
for token in tokens:
    if token not in freq:
        freq[token] = 1
    else:
        freq[token] += 1
freq_rank = {k: v for k, v in sorted(freq.items(), key=lambda x: x[1], reverse=True)}

plt.scatter(
    [math.log(i + 1) for i in range(len(freq_rank))],
    list(math.log(int(freq)) for freq in freq_rank.values()),
)
plt.xlabel("出現順位[位]", fontname="Noto Sans CJK JP")
plt.ylabel("出現頻度[回]", fontname="Noto Sans CJK JP")
plt.savefig("zip.png")
