with open('uk.txt', encoding='utf-8') as f:
    uk_text = f.read()

import re

def remove_emphasis(text):
    return re.sub(r"'{2,5}", "", text)
def remove_internal_link(text):
    pattern = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+?)\]\]")
    return pattern.sub(r"\1", text)

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
    field = remove_internal_link(remove_emphasis(field))
    value = remove_internal_link(remove_emphasis(field))
    info[field] = value

print(info)