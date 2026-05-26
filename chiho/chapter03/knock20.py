import gzip
import json
from pathlib import Path


DATA_PATH = Path(__file__).with_name("jawiki-country.json.gz")


def get_article_text(title: str = "イギリス", path: Path = DATA_PATH) -> str:
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            if article["title"] == title:
                return article["text"]

    raise ValueError(f"{title!r} の記事が見つかりませんでした")


if __name__ == "__main__":
    print(get_article_text())
