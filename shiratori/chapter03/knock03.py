import re

from knock00 import uk_txt

sections = []

for line in uk_txt.split("\n"):
    match = re.search(r"^(=+)\s*(.+?)\s*\1$", line)
    if match:
        level = len(match.group(1)) - 1
        section_name = match.group(2)

        sections.append((section_name, level))


if __name__ == "__main__":
    print(f"{section_name}: {level}")
