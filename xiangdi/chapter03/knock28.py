# 28. MediaWikiマークアップの除去
# 27の処理に加えて、テンプレートの値からMediaWikiマークアップを可能な限り除去し、
# 国の基本情報を整形せよ。

import re

with open("/Users/caitlyn/Documents/uk.txt", "r", encoding="utf-8") as f:
    text = f.read()

def remove(text):
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^|\]]+)\]\]", r"\1", text)
    
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)

    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text)
    text = re.sub(r"<ref[^>]*/>", "", text)

    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"\{\{lang\|[^|}]+\|([^}]+)\}\}", r"\1", text)

    text = re.sub(r"\{\{仮リンク\|([^|}]+).*?\}\}", r"\1", text)

    return text.strip()

template = re.search(r"\{\{基礎情報.*?\n(.*?)\n\}\}", text, re.DOTALL).group(1)
pattern = r"^\|(.*?)\s*=\s*(.*?)(?=\n\|.+?\s*=|\Z)"

basic_info = {}
for match in re.finditer(pattern, template, re.DOTALL | re.MULTILINE):
    field_name = match.group(1).strip()
    value = match.group(2).strip()
    value = remove(value)
    basic_info[field_name] = value

print(basic_info)