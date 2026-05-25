import re
from collections import Counter
import MeCab

with open('kokoro.txt', 'r', encoding='utf-8') as f:
    raw = f.read()

text = re.sub(r'《.*?》', '', raw)
text = re.sub(r'｜', '', text)

chapter_pattern = re.compile(r'^[一二三四五六七八九十百]+$', re.MULTILINE)
chapters = chapter_pattern.split(text)
chapters = [c.strip() for c in chapters if c.strip()]

tagger = MeCab.Tagger('-r /opt/homebrew/etc/mecabrc')
counter = Counter()

for chapter in chapters:
    node = tagger.parseToNode(chapter)
    while node:
        if node.surface and node.feature.split(',')[0] == '名詞':
            counter[node.surface] += 1
        node = node.next

if __name__ == "__main__":
    for word, freq in counter.most_common(20):
        print(f'{word}\t{freq}')
