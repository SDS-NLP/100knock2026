import re

with open('output_knock20.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 基礎情報テンプレートを抽出
template_match = re.search(r'\{\{基礎情報.*?\n(.+?)\n\}\}', text, re.DOTALL)

infobox = {}

if template_match:
    template_text = template_match.group(1)

    fields = re.findall(
        r'^\|\s*(.+?)\s*=\s*(.+?)(?=^\||\Z)',
        template_text,
        re.MULTILINE | re.DOTALL
    )

    for key, value in fields:
        infobox[key.strip()] = value.strip()

for key, value in infobox.items():
    print(f'【{key}】{value}')