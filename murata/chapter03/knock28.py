import json
with open('uk.txt', encoding='utf-8') as f:
    uk_text = f.read()

import re

def remove_emphasis(text):
    return re.sub(r"'{2,5}", "", text)
def remove_internal_link(text):
    pattern = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+?)\]\]")
    return pattern.sub(r"\1", text)
def clean_markup(text):
    text = re.sub(r"'{2,3}|'{5}", "", text)              # 26: 強調
    text = re.sub(r"<ref[^/]*?/>", "", text)             # ref 自己閉じ
    text = re.sub(r"<ref[^>]*?>.*?</ref>", "", text, flags=re.DOTALL)  # ref
    text = re.sub(r"\{\{lang\|[^|]+\|([^}]+)\}\}", r"\1", text)  # lang
    text = re.sub(r"\{\{[^}]+\}\}", "", text)            # その他テンプレート
    text = re.sub(r"\[\[(?:[^\]]*\|)?([^|\]]+)\]\]", r"\1", text)  # 27: 内部リンク
    text = re.sub(r"\[https?://[^ \]]+ ([^\]]+)\]", r"\1", text)  # 外部リンク（名前付き）
    text = re.sub(r"\[https?://[^\]]+\]", "", text)      # 外部リンク（URLのみ）
    text = re.sub(r"<[^>]+>", "", text)                  # その他HTMLタグ
    return text

template_pattern = re.compile(
    r"\{\{基礎情報.*?\n(.*?)\n\}\}",
    re.DOTALL 
)
m = template_pattern.search(uk_text)
template_body = m.group(1)


field_pattern = re.compile(
    r"^\|\s*([^=]+?)\s*=\s*(.+?)(?=\n\||\n$)",
    re.MULTILINE | re.DOTALL
)

info = {}
for m in field_pattern.finditer(template_body):
    field = m.group(1).strip()
    value = m.group(2).strip()
    field = clean_markup(remove_internal_link(remove_emphasis(field)))
    value = clean_markup(remove_internal_link(remove_emphasis(value)))
    info[field] = value

print(info)

with open('uk_info.json', 'w', encoding='utf-8') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
    
