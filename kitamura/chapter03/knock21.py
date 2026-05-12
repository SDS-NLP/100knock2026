import re

file = "uk_article.txt"
with open(file, "r", encoding="utf-8") as f:
    uk_text = f.readlines()

pattern = r'^\[\[Category:.*\]\]$'
for line in uk_text:
    result = re.findall(pattern, line)
    if result:
        print(result)
