import re

from knock20 import get_article_text


SECTION_PATTERN = re.compile(r"^(=+)\s*(.*?)\s*\1$")


def extract_sections(text):
    sections = []
    for line in text.splitlines():
        match = SECTION_PATTERN.match(line)
        if match:
            level = len(match.group(1)) - 1
            name = match.group(2)
            sections.append((name, level))

    return sections


if __name__ == "__main__":
    for name, level in extract_sections(get_article_text()):
        print(f"{name}: {level}")
