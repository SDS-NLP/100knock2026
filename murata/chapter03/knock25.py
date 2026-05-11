with open('uk.txt', encoding='utf-8') as f:
    uk_text = f.read()

import re
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
    info[field] = value

print(info)