import re


def extract_category_names(input_file):
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\[\[Category:([^|\]]+)", line)
            if m:
                yield m.group(1)


if __name__ == "__main__":
    input_file = "data/uk.txt"
    for name in extract_category_names(input_file):
        print(name)
