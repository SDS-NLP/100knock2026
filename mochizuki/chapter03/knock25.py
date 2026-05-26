import gzip
import json
import re

with gzip.open('jawiki-country.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            text = article['text']
            break

m = re.search(r'\{\{基礎情報[^\n]*\n(.*?)\n\}\}', text, re.DOTALL)
block = m.group(1)

info = {}
current_key = None
for line in block.splitlines():
    field = re.match(r'^\|(.+?)\s*=\s*(.*)', line)
    if field:
        current_key = field.group(1).strip()
        info[current_key] = field.group(2).strip()
    elif current_key and not line.startswith('}}'):
        info[current_key] += '\n' + line

for k, v in info.items():
    print(f'{k}: {v}')
