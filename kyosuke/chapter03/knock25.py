import re 
path = 'uk.txt'
dict = {}
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
info = re.search(r'^\{\{基礎情報 国\n(.*?)\n\}\}', text, re.MULTILINE | re.DOTALL)
if info:
        info_text = info.group(1)
matches = re.finditer(r'^\|(.+?)\s*=\s*(.*?)(?=\n\||\n\}\})', info_text, re.MULTILINE | re.DOTALL)
for match in matches:
    key = match.group(1)
    value = match.group(2)
    dict[key] = value
print(dict)