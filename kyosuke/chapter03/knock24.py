import re
path = "uk.txt"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r'\[\[ファイル:([^|]*)', line)
        if match:
            print(match.group(1))