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
    infobox_pattern = r"\{\{基礎情報 国\n(.*?)\n\}\}"
    infobox_text = re.search(infobox_pattern, uk_text, re.DOTALL).group(1)

    pattern = r"^\|(.+?)\s*=\s*(.+?)(?=\n(?:[\|\}]))"

    fields = re.findall(pattern, infobox_text + "\n}", re.MULTILINE | re.DOTALL)

    ans_dict = dict(fields)

cleaned_dict = {}

pattern = r"''+?(.+?)''+?"

for key, value in ans_dict.items():
    cleaned_value = re.sub(pattern, r"\1", value)

for key, value in cleaned_dict.items():
    print(f"{key}: {value}")