import re

with open('output_knock20.txt', 'r', encoding='utf-8') as f:
    text = f.read()

template_match = re.search(r'\{\{基礎情報\s*\S*\s*\n(.*?)\n\}\}', text, re.DOTALL)

def remove_emphasis(value: str) -> str:
    """弱い強調・強調・強い強調のマークアップを除去"""
    return re.sub(r"'{2,5}(.*?)'{2,5}", r'\1', value, flags=re.DOTALL)

infobox = {}

if template_match:
    template_text = template_match.group(1)

    fields = re.findall(
        r'^\|(.+?)\s*=\s*(.+?)(?=^\||\Z)',
        template_text,
        re.MULTILINE | re.DOTALL
    )

    for key, value in fields:
        clean_value = remove_emphasis(value.strip())
        infobox[key.strip()] = clean_value

for key, value in infobox.items():
    print(f'【{key}】{value}')