import gzip
import json

def get_uk_text(filepath):
    """Wikipediaの記事情報が格納されたJSONから「イギリス」に関する記事本文を抽出する関数"""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['title'] == 'イギリス':
                return data['text']
    return ""

if __name__ == "__main__":
    filepath = './chapter03/jawiki-country.json.gz'
    text = get_uk_text(filepath)
    print(text)