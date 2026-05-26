import re

try:
    from knock20 import get_article_text
    from knock26 import remove_emphasis_from_basic_info
except ModuleNotFoundError:
    from .knock20 import get_article_text
    from .knock26 import remove_emphasis_from_basic_info


LINK_PATTERN = re.compile(r"\[\[([^|\]]+?\|)?([^|\]]+?)\]\]")


def remove_internal_links(value):
    return LINK_PATTERN.sub(r"\2", value)


def remove_internal_links_from_basic_info(text):
    basic_info = remove_emphasis_from_basic_info(text)
    return {field_name: remove_internal_links(value) for field_name, value in basic_info.items()}


if __name__ == "__main__":
    for field_name, value in remove_internal_links_from_basic_info(get_article_text()).items():
        print(f"{field_name}: {value}")
