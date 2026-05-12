import gzip
import json
import re

with gzip.open('jawiki-country.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            for m in re.finditer(r'\[\[(?:ファイル|File):([^|\]]+)', article['text']):
                print(m.group(1))
            break
