import re
from knock20 import get_uk_text

def extract_category_lines(text):
    """カテゴリ名を宣言している行を抽出する関数"""
    return re.findall(r'^(.*\[\[Category:.*\]\].*)$', text, re.MULTILINE)

if __name__ == "__main__":
    text = get_uk_text('./chapter03/jawiki-country.json.gz')
    for line in extract_category_lines(text):
        print(line)