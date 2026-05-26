# q26.py
import re
from knock25 import extract_basic_info


def remove_emphasis(value):
    return re.sub(r"'{2,5}", "", value)


def extract_basic_info_clean(input_file):
    info = extract_basic_info(input_file)
    return {k: remove_emphasis(v) for k, v in info.items()}


if __name__ == "__main__":
    input_file = "data/uk.txt"
    info = extract_basic_info_clean(input_file)
    for k, v in info.items():
        print(f"{k}: {v}")
