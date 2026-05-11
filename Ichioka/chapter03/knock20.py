import json

def extract_article(filename: str, title: str) -> str | None:
    with open(filename) as f:
        for line in f:
            article = json.loads(line)
            if article['title'] == title:
                return article['text']
    return None

# 「イギリス」の記事を抽出
text = extract_article('jawiki-country.json', 'イギリス')

with open('output_knock20.txt', 'w', encoding='utf-8') as f:
    f.write(text)