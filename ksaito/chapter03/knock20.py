import json


input_file = "data/jawiki-country.json"
output_file = "data/uk.txt"

with open(input_file, encoding="utf-8") as f:
    for line in f:
        article = json.loads(line)
        if article["title"] != "イギリス":
            continue

        with open(output_file, "w", encoding="utf-8") as out:
            out.write(article["text"])
