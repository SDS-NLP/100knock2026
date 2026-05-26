import re
from knock20 import get_uk_text

def extract_templete(text):
    """「基礎情報」テンプレートのフィールド名と値を抽出し、辞書として格納する関数"""
    match = re.search(r'^\{\{基礎情報.*?\n(.*?)\n\}\}$', text, re.MULTILINE | re.DOTALL)
    if not match:
        return {}
    
    templete_text = match.group(1)

    fields = re.findall(r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))', templete_text, re.MULTILINE | re.DOTALL)

    return {k: v for k, v in fields}

if __name__ == "__main__":
    text = get_uk_text('./chapter03/jawiki-country.json.gz')
    for k, v in extract_templete(text).items():
        print(f"{k}: {v}")