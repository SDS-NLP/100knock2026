# 22. カテゴリ名の抽出
# 記事のカテゴリ名を（行単位ではなく名前で）抽出せよ。

import re

with open("/Users/caitlyn/Documents/category_line.txt", "r", 
               encoding="utf-8") as f:
    text = f.read()

pattern = r"\[\[Category:(.*?)(?:\|.*)?\]\]"

# (.*?): Capture any content, but as little as possible.
# (?: ... ): （non-capturing group)
# An optional | followed by any content (not captured).

names = re.findall(pattern, text)

with open("category_name.txt", "w", encoding="utf-8") as f:
    f.writelines(name + "\n" for name in names)