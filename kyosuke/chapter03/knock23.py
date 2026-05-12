import re
path = "uk.txt"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r'^(={2,})(.*?)={2,}', line)
        if match:
            level = len(match.group(1)) - 1
            print(match.group(2), level)