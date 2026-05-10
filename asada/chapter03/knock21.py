import json
from pathlib import Path

text = Path("uk.json")

with text.open("r", encoding="UTF-8") as f:
    article = json.load(f)
print("\n".join(line for line in article.split("\n") if "[[Category:" in line))
