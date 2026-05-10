from knock20 import extract_uk_text
import re

UK_text = extract_uk_text()
pattern = r"\[\[Category:.*?\]\]"
result  = re.findall(pattern, UK_text)
print(result)