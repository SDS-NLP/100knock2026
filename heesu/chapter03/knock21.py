import re
from pathlib import Path


def get_categories(file: Path):
    with open(file, "r") as fp:
        for line in fp:
            if "[[Category:" in line:
                print(line)


if __name__ == "__main__":
    get_categories(Path("イギリス.txt"))
