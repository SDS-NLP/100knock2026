import gzip
import json
import re

def solve_knock27():
    filename = "jawiki-country.json.gz"
    uk_text = ""

    with gzip.open(filename, "rt") as f:
        for line in f:
            data = json.loads(line)
            if data["title"] == "イギリス":
                uk_text = data["text"]
                break
    
    if not uk_text:
        print("記事が見つかりませんでした。")
        return
    
    infobox_pattern = r"^\{\{基礎情報 国\n(.*?)\n\}\}$"
    infobox_match = re.search(infobox_pattern, uk_text, re.MULTILINE | re.DOTALL)
    
    if not infobox_match:
        print("基礎情報が見つかりませんでした。")
        return
    
    infobox_content = infobox_match.group(1)

    field_pattern = r"^\|(.+?)\s*=\s*(.+?)(?=\n(?:[\|\}]))"
    raw_fields = re.findall(field_pattern, infobox_content + "\n}", re.MULTILINE | re.DOTALL)

    ans_dict = {}

    for key, value in raw_fields:
        cleaned_value = re.sub(r"'{2,5}", "", value)

        link_pattern = r"\[\[(?:[^|]+?\|)?([^|\]]+?)\]\]"
        cleaned_value = re.sub(link_pattern, r"\1", cleaned_value)

        ans_dict[key] = cleaned_value

        for key, value in ans_dict.items():
         print(f"{key}: {value}")