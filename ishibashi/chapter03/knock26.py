import re
from knock20 import get_uk_text
from knock25 import extract_templete

def remove_markup(text):
    """テンプレートの値からMediaWikiの強調マークアップを除去する関数"""
    return re.sub(r"'''''|'''|''", "", text)

if __name__ == "__main__":
    text = get_uk_text('./chapter03/jawiki-country.json.gz')
    templete = extract_templete(text)

    result = {}
    for k, v in templete.items():
        result[k] = remove_markup(v)

    for k, v in result.items():
        print(f"{k}: {v}")