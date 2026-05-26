import re
import math
from collections import Counter, defaultdict
import MeCab

with open('kokoro.txt', 'r', encoding='utf-8') as f:
    raw = f.read()

text = re.sub(r'《.*?》', '', raw)
text = re.sub(r'｜', '', text)

chapter_pattern = re.compile(r'^[一二三四五六七八九十百]+$', re.MULTILINE)
chapters = [c.strip() for c in chapter_pattern.split(text) if c.strip()]

tagger = MeCab.Tagger('-r /opt/homebrew/etc/mecabrc')

def extract_nouns(text):
    nouns = []
    node = tagger.parseToNode(text)
    while node:
        if node.surface and node.feature.split(',')[0] == '名詞':
            nouns.append(node.surface)
        node = node.next
    return nouns

noun_docs = [extract_nouns(c) for c in chapters]
N = len(noun_docs)

df = defaultdict(int)
for nouns in noun_docs:
    for word in set(nouns):
        df[word] += 1

best = {}
for nouns in noun_docs:
    tf = Counter(nouns)
    total = len(nouns)
    for word, count in tf.items():
        t = count / total
        i = math.log(N / df[word])
        ti = t * i
        if word not in best or ti > best[word][2]:
            best[word] = (t, i, ti)

results = [(w, *v) for w, v in best.items()]
results.sort(key=lambda x: x[3], reverse=True)

if __name__ == "__main__":
    print(f'{"単語":<10}\t{"TF":<10}\t{"IDF":<10}\tTF-IDF')
    for word, t, i, ti in results[:20]:
        print(f'{word:<10}\t{t:.6f}\t{i:.6f}\t{ti:.6f}')
