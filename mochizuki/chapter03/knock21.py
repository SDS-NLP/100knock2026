import gzip
import json

with gzip.open('jawiki-country.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            for text_line in article['text'].splitlines():
                if text_line.startswith('[[Category:'):
                    print(text_line)
            break
