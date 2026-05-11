with open('uk.txt', encoding='utf-8') as f:
    uk_text = f.read()

import re
pattern = re.compile(r"\[\[Category:(.+?)(?:\|.*)?\]\]")
for line in uk_text.split("\n"):
    m = pattern.search(line)
    if m:
        print(m.group(1))