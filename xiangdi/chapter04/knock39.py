# 39. Zipfの法則
# コーパスにおける単語の出現頻度順位を横軸、その出現頻度を縦軸として、両対数グラフをプロットせよ。
import MeCab
from collections import Counter
import matplotlib.pyplot as plt
import re

with open("/Users/caitlyn/Downloads/kokoro.txt", encoding="utf-8") as f:
    text = f.read()

text = re.sub(r"《.*?》", "", text)
text = re.sub(r"［＃.*?］", "", text)
text = text.replace("｜", "")
text = text.replace("\r", "")

tagger = MeCab.Tagger()
node = tagger.parseToNode(text)

words = []

while node:
    features = node.feature.split(",")

    if node.surface and features[0] != "補助記号":
        words.append(node.surface)

    node = node.next

counter = Counter(words)

freqs = [freq for word, freq in counter.most_common()]
ranks = range(1, len(freqs) + 1)

plt.plot(ranks, freqs)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Zipf's Law")
plt.grid(True)
plt.show()
