import re
from knock20 import get_uk_text

def extract_media_files(text):
    """記事から参照されているメディアファイルを抽出する関数"""
    return re.findall(r'\[\[(?:ファイル|File|画像):(.*?)(?:\|.*)?\]\]', text)

if __name__ == "__main__":
    text = get_uk_text('./chapter03/jawiki-country.json.gz')
    for file in extract_media_files(text):
        print(file)