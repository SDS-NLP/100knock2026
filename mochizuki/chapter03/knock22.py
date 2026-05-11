import gzip
import json
import re

with gzip.open('jawiki-country.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            for text_line in article['text'].splitlines():
                m = re.match(r'\[\[Category:([^|\]]+)', text_line)
                if m:
                    print(m.group(1))
            break
