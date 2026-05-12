import re

pattern = r"\[\[(?:File|ファイル):([^|\]]+)(?:\|.*)?\]\]"

with open("イギリス.txt", "r") as fp:
    files = re.findall(pattern, fp.read())

for file_name in files:
    print(file_name)
