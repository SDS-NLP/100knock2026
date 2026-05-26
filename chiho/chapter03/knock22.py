import re

from knock20 import get_article_text
from knock21 import extract_category_lines


CATEGORY_PATTERN = re.compile(r"\[\[Category:(.*?)(?:\|.*)?\]\]")


def extract_category_names(text):
    category_names = []
    for line in extract_category_lines(text):
        match = CATEGORY_PATTERN.search(line)
        if match:
            category_names.append(match.group(1))

    return category_names


if __name__ == "__main__":
    for name in extract_category_names(get_article_text()):
        print(name)
