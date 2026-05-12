import gzip
import json
import re

def solve_knock28():
    filename = "jawiki-country.json.gz"
    uk_text = ""

    with gzip.open(filename, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["title"] == "イギリス":
                uk_text = data["text"]
                break

    infobox_pattern = r"^\{\{基礎情報 国\n(.*?)\n\}\}$"
    infobox_match = re.search(infobox_pattern, uk_text, re.MULTILINE | re.DOTALL)
    if not infobox_match: return
    
    infobox_content = infobox_match.group(1)
    field_pattern = r"^\|(.+?)\s*=\s*(.+?)(?=\n(?:[\|\}]))"
    raw_fields = re.findall(field_pattern, infobox_content + "\n}", re.MULTILINE | re.DOTALL)

    ans_dict = {}

    for key, value in raw_fields:
        
        value = re.sub(r"'{2,5}", "", value)
        
        value = re.sub(r"\[\[(?:[^|]+?\|)?([^|\]]+?)\]\]", r"\1", value)

        value = re.sub(r"\{\{(?:[^|]+?\|)*([^|]+?)\}\}", r"\1", value)

        value = re.sub(r"\[http[^\s]*?\s+(.*?)\]", r"\1", value)

        value = re.sub(r"<[^>]+?>", "", value)

        ans_dict[key] = value

    for key, value in ans_dict.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    solve_knock28()