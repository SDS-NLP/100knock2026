import re

file = "uk_article.txt"

with open(file, "r", encoding="utf-8") as f:
    content = f.read()

pattern = r"\[\[ファイル:(.*?)\]\]"

result = re.findall(pattern, content)

for name in result:
    print(name)