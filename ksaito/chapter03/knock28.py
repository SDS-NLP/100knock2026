# knock28.py
import re

from knock27 import extract_basic_info_clean as extract_v27


def remove_file_references(value):
    return re.sub(r"\[\[(?:ファイル|File):[^\]]+\]\]", "", value)

def remove_templates(value):
    while True:
        new_value = re.sub(r"\{\{[^{}]*\}\}", "", value)
        if new_value == value:
            return new_value
        value = new_value

def remove_external_links(value):
    return re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", value)

def remove_ref_tags(value):
    value = re.sub(r"<ref[^>]*?/>", "", value)
    value = re.sub(r"<ref[^>]*>.*?</ref>", "", value, flags=re.DOTALL)
    value = re.sub(r"<references/>", "", value)
    return value

def remove_br_tags(value):
    return re.sub(r"<br\s*/?>", "", value)

def clean(value):
    value = remove_file_references(value)
    value = remove_ref_tags(value)
    value = remove_templates(value)
    value = remove_external_links(value)
    value = remove_br_tags(value)
    return value.strip()

def extract_basic_info_clean(input_file):
    info = extract_v27(input_file)
    return {k: clean(v) for k, v in info.items()}

if __name__ == "__main__":
    input_file = "data/uk.txt"
    info = extract_basic_info_clean(input_file)
    for k, v in info.items():
        print(f"{k}: {v}")
