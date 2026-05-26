import re
import MeCab
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

with open('kokoro.txt', 'r', encoding='utf-8-sig') as f:
    content = f.read()

content_clean = re.sub(r'《[^》]*》', '', content)
content_clean = re.sub(r'[|]', '', content_clean)
content_clean = content_clean.replace('\r\n', '\n')

pattern = r'(?:^|\n)[一二三四五六七八九十百千万]+\n'
chapter = re.split(pattern, content_clean)
chapters = [ch.strip() for ch in chapter if ch.strip() != '']

tagger = MeCab.Tagger()

total_chapters = len(chapters)

total_words_list = []
for chapter in chapters:
    node = tagger.parseToNode(chapter)
    while node:
        if node.surface != "":
            features = node.feature.split(',')
            if features[0] != '補助記号':
                base = features[7] if len(features) > 7 and features[7] != '*' else node.surface
                total_words_list.append(base)
        node = node.next
words_count = Counter(total_words_list)

sorted_counts = [count for word, count in words_count.most_common()]

x = np.array(range(1, len(sorted_counts) + 1)) 
y = np.array(sorted_counts)
plt.loglog(x, y)

plt.title("Zipf's Law")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()