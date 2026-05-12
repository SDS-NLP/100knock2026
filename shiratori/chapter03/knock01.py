import re

from knock00 import uk_txt


categories = []

for line in uk_txt.split("\n"):
    if re.search(r"\[\[Category:.*\]\]", line):
        categories.append(line)

if __name__ == "__main__":
    print(line)
