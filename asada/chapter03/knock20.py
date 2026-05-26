import json
from pathlib import Path

text = Path("jawiki-country.json")

with text.open(encoding="UTF-8") as f:
    articles = [json.loads(line) for line in f]
    uk = next(article["text"] for article in articles if article["title"] == "イギリス")

with Path("uk.json").open("w", encoding="UTF-8") as f:
    json.dump(uk, f)

print(uk)
