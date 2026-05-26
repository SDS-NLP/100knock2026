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
    pattern = r"^(=+)\s*(.*?)\s*(=+)$"

    sections = re.findall(pattern, uk_text, re.MULTILINE)

    for match in sections:
        equals = match[0]    
        name = match[1]       
        level = len(equals) - 1

        print(f"{name}: {level}")