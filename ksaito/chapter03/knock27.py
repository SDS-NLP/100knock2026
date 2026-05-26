import re
from knock26 import extract_basic_info_clean as extract_v26


def remove_internal_links(value):
    pattern = r"\[\[(?!ファイル:|File:)(?:[^|\]]*\|)*([^|\]]+)\]\]"
    return re.sub(pattern, r"\1", value)


def extract_basic_info_clean(input_file):
    info = extract_v26(input_file)
    return {k: remove_internal_links(v) for k, v in info.items()}


if __name__ == "__main__":
    input_file = "data/uk.txt"
    info = extract_basic_info_clean(input_file)
    for k, v in info.items():
        print(f"{k}: {v}")
