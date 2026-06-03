import MeCab
from collections import Counter

with open("kokoro.txt", "r", encoding="utf-8") as f:
    text = f.read()

tagger = MeCab.Tagger()
words = []

for line in text.split('\n'):
    if not line:
        continue
    node = tagger.parseToNode(line)
    while node:
        if node.surface == '':
            node = node.next
            continue
            
        features = node.feature.split(',')
        if features[0] != '補助記号':
            base = features[7] if len(features) > 7 and features[7] != '*' else node.surface
            words.append(base)
            
        node = node.next

word_counts = Counter(words)
top_20 = word_counts.most_common(20)
for word, count in top_20:
    print(f"{word}: {count}回")