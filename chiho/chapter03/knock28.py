import re

try:
    from knock20 import get_article_text
    from knock25 import extract_basic_info
except ModuleNotFoundError:
    from .knock20 import get_article_text
    from .knock25 import extract_basic_info


REF_PATTERN = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)
BR_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
TAG_PATTERN = re.compile(r"</?[^>]+>")

EXTERNAL_LINK_WITH_LABEL_PATTERN = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")
EXTERNAL_LINK_ONLY_PATTERN = re.compile(r"\[https?://[^\]]+\]")

FILE_LINK_PATTERN = re.compile(
    r"\[\[(?:ファイル|File|画像|Image):([^|\]]+)(?:\|[^\]]*)?\]\]",
    re.IGNORECASE,
)

INTERNAL_LINK_PATTERN = re.compile(r"\[\[(?:[^|\]]*\|)*([^|\]]+)\]\]")

LANG_TEMPLATE_PATTERN = re.compile(r"\{\{lang\|[^|{}]+\|([^{}]+)\}\}", re.IGNORECASE)
TEMP_LINK_PATTERN = re.compile(r"\{\{仮リンク\|([^|{}]+)\|[^{}]*\}\}")
ICON_TEMPLATE_PATTERN = re.compile(r"\{\{[a-z]{2,3}\s+icon\}\}", re.IGNORECASE)
SIMPLE_TEMPLATE_PATTERN = re.compile(r"\{\{[^{}|]+\|([^{}]*)\}\}")
EMPTY_TEMPLATE_PATTERN = re.compile(r"\{\{0\}\}")

WHITESPACE_PATTERN = re.compile(r"[ \t]+")


def remove_references(text: str) -> str:
    """脚注を除去する。"""
    return REF_PATTERN.sub("", text)


def replace_br_with_newline(text: str) -> str:
    """<br> や <br /> を改行に変換する。"""
    return BR_PATTERN.sub("\n", text)


def remove_html_tags(text: str) -> str:
    """残った HTML タグを除去する。"""
    return TAG_PATTERN.sub("", text)


def remove_external_links(text: str) -> str:
    """外部リンクを表示文字列だけにする。"""
    text = EXTERNAL_LINK_WITH_LABEL_PATTERN.sub(r"\1", text)
    text = EXTERNAL_LINK_ONLY_PATTERN.sub("", text)
    return text


def remove_file_links(text: str) -> str:
    """
    ファイル・画像リンクからファイル名だけを残す。

    例:
    [[ファイル:Example.svg|thumb|説明]] -> Example.svg
    """
    return FILE_LINK_PATTERN.sub(r"\1", text)


def remove_internal_links(text: str) -> str:
    """
    内部リンクを表示文字列だけにする。

    例:
    [[ロンドン]] -> ロンドン
    [[ロンドン|London]] -> London
    [[A|B|C]] -> C
    """
    return INTERNAL_LINK_PATTERN.sub(r"\1", text)


def remove_templates_once(text: str) -> str:
    """単純なテンプレートを1回分だけ処理する。"""
    text = EMPTY_TEMPLATE_PATTERN.sub("", text)
    text = LANG_TEMPLATE_PATTERN.sub(r"\1", text)
    text = TEMP_LINK_PATTERN.sub(r"\1", text)
    text = ICON_TEMPLATE_PATTERN.sub("", text)
    text = SIMPLE_TEMPLATE_PATTERN.sub(r"\1", text)
    return text


def remove_templates(text: str) -> str:
    """単純なテンプレートを、変化がなくなるまで除去する。"""
    previous = None
    while text != previous:
        previous = text
        text = remove_templates_once(text)
    return text


def normalize_whitespace(text: str) -> str:
    """余分な空白や空行を整える。"""
    lines = [
        WHITESPACE_PATTERN.sub(" ", line).strip()
        for line in text.splitlines()
    ]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def clean_markup(text: str) -> str:
    """MediaWiki マークアップを可能な範囲で除去する。"""
    text = remove_references(text)
    text = replace_br_with_newline(text)
    text = remove_external_links(text)

    previous = None
    while text != previous:
        previous = text
        text = remove_file_links(text)
        text = remove_templates(text)
        text = remove_internal_links(text)

    text = remove_html_tags(text)
    text = normalize_whitespace(text)
    return text


def clean_basic_info(text: str) -> dict[str, str]:
    """基礎情報テンプレートを抽出し、各値からマークアップを除去する。"""
    basic_info = extract_basic_info(text)
    return {
        field_name: clean_markup(value)
        for field_name, value in basic_info.items()
    }


if __name__ == "__main__":
    for field_name, value in clean_basic_info(get_article_text()).items():
        print(f"{field_name}: {value}")
