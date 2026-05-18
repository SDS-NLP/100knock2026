import MeCab
from collections import Counter

with open("/Users/caitlyn/Downloads/kokoro.txt", "r", encoding="utf-8") as f:
    text = f.read()

tagger = MeCab.Tagger()
node = tagger.parseToNode(text)

nouns = []

while node:
    features = node.feature.split(",")

    if features[0] == "名詞":
        nouns.append(node.surface)

    node = node.next

counter = Counter(nouns)

for word, freq in counter.most_common(20):
    print(word, freq)