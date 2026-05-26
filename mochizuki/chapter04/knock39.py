import re
from collections import Counter
import MeCab
import matplotlib.pyplot as plt

with open('kokoro.txt', 'r', encoding='utf-8') as f:
    raw = f.read()

text = re.sub(r'《.*?》', '', raw)
text = re.sub(r'｜', '', text)

tagger = MeCab.Tagger('-r /opt/homebrew/etc/mecabrc')
counter = Counter()
node = tagger.parseToNode(text)
while node:
    if node.surface:
        counter[node.surface] += 1
    node = node.next

freqs = [freq for _, freq in counter.most_common()]
ranks = range(1, len(freqs) + 1)

if __name__ == "__main__":
    plt.figure()
    plt.loglog(ranks, freqs)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title("Zipf's Law")
    plt.savefig('knock39.png')
    plt.show()
