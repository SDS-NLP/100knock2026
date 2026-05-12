import re


def extract_file_references(input_file):
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            for m in re.finditer(r"\[\[(?:File|ファイル):([^|\]]+)", line):
                yield m.group(1)


if __name__ == "__main__":
    input_file = "data/uk.txt"
    for name in extract_file_references(input_file):
        print(name)
