import re
from knock20 import get_uk_text
from knock25 import extract_templete
from knock26 import remove_markup
from knock27 import remove_internal_links

def remove_mediawiki_markup(text):
    """テンプレートの値からMediaWikiマークアップを可能な限り除去する関数"""
    text = re.sub(r'', '', text)
    text = re.sub(r'<.+?>', '', text)
    text = re.sub(r'\[https?://(?:[^\s\]]+)(?:\s([^\]]+))?\]', r'\1', text)
    text = re.sub(r'(?<!\[)https?://[^\s]+', '', text)
    text = re.sub(r'\[\[(?:ファイル|File|画像):(?:[^|]*?\|)*?([^|]*?)\]\]', r'\1', text)
    text = re.sub(r'\{\{(?:[^|]*?\|)*?([^|]*?)\}\}', r'\1', text)
    text = re.sub(r'^[*\#;:]+\s*', '', text, flags=re.MULTILINE)

    return text.strip()

if __name__ == "__main__":
    text = get_uk_text('./chapter03/jawiki-country.json.gz')
    templete = extract_templete(text)

    result = {}
    for k, v in templete.items():
        v = remove_markup(v)
        v = remove_internal_links(v)
        v = remove_mediawiki_markup(v)
        result[k] = v

    for k, v in result.items():
        print(f"{k}: {v}")