import re

with open('output_knock20.txt', 'r', encoding='utf-8') as f:
    text = f.read()

template_match = re.search(r'\{\{基礎情報\s*\S*\s*\n(.*?)\n\}\}', text, re.DOTALL)

def remove_markup(value: str) -> str:

    # 1: 強調マークアップの除去
    value = re.sub(r"'{2,5}(.*?)'{2,5}", r'\1', value, flags=re.DOTALL)

    # 2: 内部リンク
    value = re.sub(r'\[\[[^\]]+\|([^\]]+)\]\]', r'\1', value)

    # 3: 内部リンク
    value = re.sub(r'\[\[([^\]|#]+)[^\]]*\]\]', r'\1', value)

    # 4: ファイル
    value = re.sub(r'\[\[(?:ファイル|File):[^\]]*\|([^\]]+)\]\]', r'\1', value)

    # 5: 外部リンク
    value = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', value)

    # 6: 外部リンク
    value = re.sub(r'\[https?://\S+\]', '', value)

    # 7: カテゴリ
    value = re.sub(r'\[\[Category:[^\]]*\]\]', '', value)

    # 8: コメントアウト 
    value = re.sub(r'<!--.*?-->', '', value, flags=re.DOTALL)

    # 9: 見出し
    value = re.sub(r'={2,6}\s*(.+?)\s*={2,6}', r'\1', value)

    # 10: 箇条書き記号（行頭の * # ; : ）
    value = re.sub(r'^\s*[*#;:]+\s*', '', value, flags=re.MULTILINE)

    # 11: 水平線
    value = re.sub(r'^----$', '', value, flags=re.MULTILINE)

    return value

infobox = {}

if template_match:
    template_text = template_match.group(1)

    fields = re.findall(
        r'^\|(.+?)\s*=\s*(.+?)(?=^\||\Z)',
        template_text,
        re.MULTILINE | re.DOTALL
    )

    for key, value in fields:
        clean_value = remove_markup(value.strip())
        infobox[key.strip()] = clean_value

for key, value in infobox.items():
    print(f'【{key}】{value}')