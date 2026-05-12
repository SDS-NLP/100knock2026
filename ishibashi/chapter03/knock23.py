import re
from knock20 import get_uk_text

def extract_sections(text):
    """セクション名とそのレベルを抽出する関数"""
    sections = re.findall(r'^(=+)\s*(.*?)\s*\1$', text, re.MULTILINE)
    return [(name, len(level) - 1) for level, name in sections]

if __name__ == "__main__":
    text = get_uk_text('./chapter03/jawiki-country.json.gz')
    for name, level in extract_sections(text):
        print(f"レベル{level}: {name}")