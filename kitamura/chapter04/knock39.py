import re
import MeCab
from collections import Counter
import matplotlib.pyplot as plt


file = "kokoro.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()

pattern = r"(《.*?》)"
text = re.sub(pattern, "", text)

tagger = MeCab.Tagger()
node = tagger.parseToNode(text)

count = []
while node:
    features = node.feature.split(",")

    try:
        base_form = features[7]
        # もし原形がアスタリスク（データなし）なら、そのままの文字を使う
        if base_form == "*":
            base_form = node.surface
    except IndexError:
        base_form = node.surface

    count.append(base_form)

    node = node.next

word_count = Counter(count)
freq = sorted(word_count.values(), reverse=True)

ranks = range(1, len(freq)+1)

plt.figure(figsize=(8,6))
plt.scatter(ranks, freq, s=10, alpha=0.5)

plt.xscale('log')
plt.yscale('log')

plt.xlabel("Rank")
plt.ylabel("Frequency")

plt.show()