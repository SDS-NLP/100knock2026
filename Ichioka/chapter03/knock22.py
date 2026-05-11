import re

with open('output_knock20.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# [[Category:カテゴリ名]] からカテゴリ名だけを抽出
categories = re.findall(r'\[\[Category:(.+?)\]\]', text)

for cat in categories:
    print(cat)