import json

filename = "shiratori/chapter03/jawiki-country.json"

with open(filename, "r") as f:
    for line in f:
        article = json.loads(line)

        if article["title"] == "イギリス":
            uk_txt = article["text"]

            break
