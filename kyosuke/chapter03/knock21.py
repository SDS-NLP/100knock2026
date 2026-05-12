import re
path = "uk.txt"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if re.search(r'\[\[Category:.*\]\]', line):
            print(line, end="")