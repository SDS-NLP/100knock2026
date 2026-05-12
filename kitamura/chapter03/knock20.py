import json
file = "jawiki-country.json"

with open(file, "r", encoding = "utf-8") as f:
    for line in f:
        article = json.loads(line)

        if article["title"] == "イギリス":
            uk_text = article["text"]
            break

print(uk_text)

output_name = "uk_article.txt"

with open(output_name, "w", encoding="utf-8") as output:
    output.writelines(uk_text)