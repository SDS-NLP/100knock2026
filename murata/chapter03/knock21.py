with open('uk.txt', encoding='utf-8') as f:
    uk_text = f.read()

import re
for line in uk_text.split():
    if re.search(r"\[\[Category:", line):
        print(line)