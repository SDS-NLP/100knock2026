import re
from knock20 import get_uk_text
from knock25 import extract_templete
from knock26 import remove_markup

def remove_internal_links(text):
    """テンプレートの値からMediaWikiの内部リンクマークアップを除去する関数"""
    return re.sub(r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]', r'\1', text)

if __name__ == "__main__":
    text = get_uk_text('./chapter03/jawiki-country.json.gz')
    templete = extract_templete(text)

    result = {}
    for k, v in templete.items():
        cleaned_v = remove_markup(v)
        result[k] = remove_internal_links(cleaned_v)

    for k, v in result.items():
        print(f"{k}: {v}")