import re
import MeCab
from collections import Counter

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

    if features[0] == "名詞":
        count.append(node.surface)


    node = node.next

word_count = Counter(count)

for word, count in word_count.most_common(20):
    print(f"{word}:{count}")