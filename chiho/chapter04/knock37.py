from collections import Counter
from pathlib import Path
import re

import MeCab


def clean_text(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = re.sub(r"《[^》]+》", "", text)
    text = re.sub(r"［＃.*?］", "", text)
    text = re.sub(r"^.+?｜", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[一二三四五六七八九十]+$", "", text, flags=re.MULTILINE)
    return text


file_path = Path(__file__).with_name("kokoro.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text = clean_text(f.read())

mecab = MeCab.Tagger(r"-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/ipadic")
node = mecab.parseToNode(text)

all_noun_counter = Counter()
independent_noun_counter = Counter()

while node:
    if node.surface:
        features = node.feature.split(",")
        if features[0] == "名詞":
            all_noun_counter[node.surface] += 1
            if features[1] != "非自立":
                independent_noun_counter[node.surface] += 1
    node = node.next

print("名詞（非自立を含む）")
for word, count in all_noun_counter.most_common(20):
    print(f"{word}\t{count}")

print()
print("名詞（非自立を除く）")
for word, count in independent_noun_counter.most_common(20):
    print(f"{word}\t{count}")
