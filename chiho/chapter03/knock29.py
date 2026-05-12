import json
import ssl
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from knock20 import get_article_text
    from knock28 import clean_basic_info
except ModuleNotFoundError:
    from .knock20 import get_article_text
    from .knock28 import clean_basic_info


API_ENDPOINT = "https://ja.wikipedia.org/w/api.php"


def get_flag_image_name(text):
    basic_info = clean_basic_info(text)
    return basic_info["国旗画像"]


def get_image_url(file_name):
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": f"File:{file_name}",
        "iiprop": "url",
    }
    url = f"{API_ENDPOINT}?{urlencode(params)}"
    ssl_context = ssl._create_unverified_context()
    request = Request(url, headers={"User-Agent": "NLP100Knock/1.0"})

    with urlopen(request, context=ssl_context) as response:
        data = json.load(response)

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    return page["imageinfo"][0]["url"]


if __name__ == "__main__":
    flag_image_name = get_flag_image_name(get_article_text())
    print(get_image_url(flag_image_name))
