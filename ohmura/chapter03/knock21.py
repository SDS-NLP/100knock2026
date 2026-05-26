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
    lines = uk_text.split("\n")
    category_lines = []
    
    for line in lines:
        if re.search(r"\[\[Category:", line):
            category_lines.append(line)

  
    for line in category_lines:
        print(line)