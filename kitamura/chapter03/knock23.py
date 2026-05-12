import re
file = "uk_article.txt"

with open(file, "r", encoding="utf-8") as f:
    content = f.read()

pattern = r"(\={2,})\s*(.*?)\s*(\={2,})"

result = re.findall(pattern, content)

for label in result:
    level1, name, level2 = label
    level = len(level1)-1
    print(f'{name}:{level}')

