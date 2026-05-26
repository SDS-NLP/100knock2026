import re

try:
    from knock20 import get_article_text
except ModuleNotFoundError:
    from .knock20 import get_article_text


BASIC_INFO_PATTERN = re.compile(r"^\{\{基礎情報.*?^\}\}$", re.MULTILINE | re.DOTALL)
FIELD_PATTERN = re.compile(r"^\|(.*?)\s*=\s*(.*?)(?=^\|.*?\s*=|\n^\}\}$)", re.MULTILINE | re.DOTALL)


def extract_basic_info(text):
    match = BASIC_INFO_PATTERN.search(text)
    if not match:
        return {}

    basic_info = {}
    for field_match in FIELD_PATTERN.finditer(match.group()):
        field_name = field_match.group(1).strip()
        value = field_match.group(2).strip()
        basic_info[field_name] = value

    return basic_info


if __name__ == "__main__":
    for field_name, value in extract_basic_info(get_article_text()).items():
        print(f"{field_name}: {value}")
