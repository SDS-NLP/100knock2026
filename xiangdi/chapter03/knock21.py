# 21. カテゴリ名を含む行を抽出
# 記事中でカテゴリ名を宣言している行を抽出せよ。

import re

result = []

with open("/Users/caitlyn/Documents/uk.txt", "r", 
               encoding="utf-8") as f:
    for line in f:
        if re.search(r"\[\[Category:", line):
            result.append(line)

with open("category_line.txt", "w", encoding="utf-8") as output:
    output.writelines(result)