import re

def extract_category_line(input_file: str):
    with open(input_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if re.search(r"^\[\[Category:.*\]\]$", line):
                yield line

if __name__ == "__main__":
    input_file = "data/uk.txt"
    for line in extract_category_line(input_file):
        print(line)
