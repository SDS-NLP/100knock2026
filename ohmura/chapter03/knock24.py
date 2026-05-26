import gzip
import json
import re

filename = "jawiki-country.json.gz"

def extract_uk_text():
    with gzip.open(filename, "rt") as f:
        for line in f:
            data = json.loads(line)

            if data["title"] == "イギリス":
                return data["text"]
            
uk_text = extract_uk_text()

if uk_text:
    pattern = r"(?:File|ファイル):(.+?)(?:\||\]\])"

    media_files = re.findall(pattern, uk_text)

    for file_name in media_files:
        print(file_name)