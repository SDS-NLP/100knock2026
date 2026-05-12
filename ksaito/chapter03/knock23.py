import re


def extract_sections(input_file):
    with open(input_file, encoding="utf-8") as f:
        text = f.read()

    sections = []
    for m in re.finditer(r"^(=+)\s*(.+?)\s*\1$", text, re.MULTILINE):
        name = m.group(2)
        level = len(m.group(1)) - 1
        sections.append((name, level))
        
    return sections


if __name__ == "__main__":
    input_file = "data/uk.txt"
    for name, level in extract_sections(input_file):
        print(name, level)
