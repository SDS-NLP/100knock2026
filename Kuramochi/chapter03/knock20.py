import json
from pathlib import Path


def extract_uk_text() -> str:

    # Path(__file__).resolve().parentで指定したファイルの絶対パスを入手
    json_path = Path(__file__).resolve().parent / "jawiki-country.json"

    with json_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if json.loads(line).get("title") == "イギリス":
            UK_text = json.loads(line).get("text")
            return UK_text
    
    return ""


if __name__ == "__main__":
    UK_text = extract_uk_text()
    print(UK_text)
