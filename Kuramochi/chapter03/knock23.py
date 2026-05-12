from knock20 import extract_uk_text
import re
import collections

UK_text = extract_uk_text()

pattern = r"(={2,4}.*?={2,4})"
result  = re.findall(pattern, UK_text)
print(result)
section = {}

for text in result:
    c1   = collections.Counter(text)
    c2   = int(c1['=']/2) - 1        # レベルの計算
    text = text.replace('=', '')
    section[text] = c2

print(section)