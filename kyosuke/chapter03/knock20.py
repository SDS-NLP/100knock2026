import json
import gzip

path = "jawiki-country.json.gz"
with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data["title"] == "イギリス":
            print(data["text"])
            with open("uk.txt", "w", encoding="utf-8") as ff:
                   ff.write(data["text"])
            break
