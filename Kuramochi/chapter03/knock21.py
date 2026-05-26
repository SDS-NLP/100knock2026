from knock20 import extract_uk_text
import re

UK_text = extract_uk_text()
pattern = re.compile(r"\[\[Category:.*?\]\]")
result  = pattern.findall(UK_text)
print(result)