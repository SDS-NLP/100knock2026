import gzip
import json
import re

with gzip.open('jawiki-country.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            for text_line in article['text'].splitlines():
                m = re.match(r'^(={2,})\s*(.+?)\s*\1$', text_line)
                if m:
                    level = len(m.group(1)) - 1
                    print(f'{m.group(2)} {level}')
            break
