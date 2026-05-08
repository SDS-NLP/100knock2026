# 25. テンプレートの抽出
# 記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し、
# 辞書オブジェクトとして格納せよ。

import re

with open("/Users/caitlyn/Documents/uk.txt", "r", encoding="utf-8") as f:
    text = f.read()

template = re.search(r"\{\{基礎情報.*?\n(.*?)\n\}\}", text, re.DOTALL).group(1)
pattern = r"^\|(.*?)\s*=\s*(.*?)(?=\n\|.+?\s*=|\Z)"

# DOTALL: The dot (.) matches everything.

basic_info = {}
for match in re.finditer(pattern, template, re.DOTALL | re.MULTILINE):
    field_name = match.group(1)
    value = match.group(2)
    basic_info[field_name] = value

print(basic_info)