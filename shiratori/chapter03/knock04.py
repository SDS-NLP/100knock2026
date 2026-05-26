import re

from knock00 import uk_txt

files = []


for line in uk_txt.split("\n"):
    match = re.search(r"\[\[ファイル:(.+?)(?:\||\])", line)

    if match:
        file_name = match.group(1)
        files.append(file_name)

if __name__ == "__main__":
    print(file_name)
