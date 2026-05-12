import re

from knock20 import get_article_text


MEDIA_FILE_PATTERN = re.compile(r"\[\[(?:ファイル|File):(.*?)(?:\||\]\])")


def extract_media_files(text):
    return MEDIA_FILE_PATTERN.findall(text)


if __name__ == "__main__":
    for media_file in extract_media_files(get_article_text()):
        print(media_file)
