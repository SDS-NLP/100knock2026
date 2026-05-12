with open('uk.txt', encoding='utf-8') as f:
    uk_text = f.read()

import re
pattern = re.compile(r"\[\[(?:ファイル|File):(.+?)(?:\|.*)?\]\]")
for m in pattern.finditer(uk_text):
    print(m.group(1))