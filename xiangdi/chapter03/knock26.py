# 26. 強調マークアップの除去
# 25の処理時に、テンプレートの値からMediaWikiの強調マークアップ
# （弱い強調、強調、強い強調のすべて）を除去してテキストに変換せよ（参考: マークアップ早見表）。

import re

with open("/Users/caitlyn/Documents/uk.txt", "r", encoding="utf-8") as f:
    text = f.read()

def remove(text):
    return re.sub(r"'{2,5}", "", text)

template = re.search(r"\{\{基礎情報.*?\n(.*?)\n\}\}", text, re.DOTALL).group(1)
pattern = r"^\|(.*?)\s*=\s*(.*?)(?=\n\|.+?\s*=|\Z)"

basic_info = {}
for match in re.finditer(pattern, template, re.DOTALL | re.MULTILINE):
    field_name = match.group(1).strip()
    value = match.group(2).strip()
    value = remove(value)
    basic_info[field_name] = value

print(basic_info)