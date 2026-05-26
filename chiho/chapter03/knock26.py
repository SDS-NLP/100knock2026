import re

try:
    from knock20 import get_article_text
    from knock25 import extract_basic_info
except ModuleNotFoundError:
    from .knock20 import get_article_text
    from .knock25 import extract_basic_info


EMPHASIS_PATTERN = re.compile(r"'{2,5}")


def remove_emphasis(value):
    return EMPHASIS_PATTERN.sub("", value)


def remove_emphasis_from_basic_info(text):
    basic_info = extract_basic_info(text)
    return {field_name: remove_emphasis(value) for field_name, value in basic_info.items()}


if __name__ == "__main__":
    for field_name, value in remove_emphasis_from_basic_info(get_article_text()).items():
        print(f"{field_name}: {value}")
