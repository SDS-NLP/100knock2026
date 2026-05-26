from knock20 import get_article_text


def extract_category_lines(text):
    return [line for line in text.splitlines() if line.startswith("[[Category:")]


if __name__ == "__main__":
    for line in extract_category_lines(get_article_text()):
        print(line)
