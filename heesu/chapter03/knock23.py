import re

pattern = r"^(={2,})\s*(.+?)\s*\1$"

with open("イギリス.txt", "r") as fp:
    sections = re.findall(pattern, fp.read(), re.MULTILINE)

for symbols, name in sections:
    level = len(symbols) - 1
    print(f"Level {level}: {name}")
