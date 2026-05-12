import re

from knock00 import uk_txt

categories = []

for line in uk_txt.split("\n"):
    match = re.search(r"\[\[Category:(.*)\]\]", line)
    if match:
        categories.append(match.group(1))

if __name__ == "__main__":
    print(match.group(1))
