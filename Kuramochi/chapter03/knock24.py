import re
from knock20 import extract_uk_text

UK_text = extract_uk_text()
pattern = r'\[\[ファイル:(.*?)(?:\||\])'
result  = re.findall(pattern, UK_text)
print(result)
