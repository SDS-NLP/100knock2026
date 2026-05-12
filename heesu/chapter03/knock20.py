import json
from pathlib import Path


def extract_article(title: str) -> Path:
    with open("jawiki-country.json", "r") as fp:
        for line in fp:
            jsonline = json.loads(line)

            if jsonline["title"] == title:
                out_path = Path(f"{title}.txt")
                with open(out_path, "w") as fp:
                    fp.write(jsonline["text"])
                    return out_path


if __name__ == "__main__":
    extract_article("イギリス")
