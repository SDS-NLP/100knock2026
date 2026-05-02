import re

with open('output_knock20.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# [[ファイル:〜]] または [[File:〜]] を抽出
media_files = re.findall(r'\[\[(?:ファイル|File):(.+?)(?:\|.*)?\]\]', text)

for media in media_files:
    print(media)