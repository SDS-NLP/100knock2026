import re
from pathlib import Path


def get_categories(file: Path):
    pattern = r"\[\[Category:([^|\]]+)(?:\|.*)?\]\]"
    with open(file, "r") as fp:
        categories = re.findall(pattern, fp.read())

    return categories


if __name__ == "__main__":
    print("\n".join(get_categories(Path("イギリス.txt"))))
