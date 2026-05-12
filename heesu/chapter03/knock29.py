import re
import requests


template_pattern = r"^\{\{基礎情報.*?$(.*?)^\}\}"

with open("イギリス.txt", "r") as fp:
    template_match = re.search(template_pattern, fp.read(), re.DOTALL | re.MULTILINE)

infobox_dict = {}

if template_match:
    infobox_content = template_match.group(1)

    field_pattern = r"^\|(.+?)\s*=\s*(.+?)(?=\n\||^$|^\}\})"
    fields = re.findall(field_pattern, infobox_content, re.MULTILINE | re.DOTALL)

    for key, value in fields:
        infobox_dict[key.strip()] = value.strip()


def clean_mediawiki_markup(text):
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s?([^\]]*)\]", r"\1", text)
    while re.search(r"\{\{[^|}]+\|([^|}]+)\}\}", text):
        text = re.sub(r"\{\{[^|}]+\|([^|}]+)\}\}", r"\1", text)
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)

    return text.strip()


cleaned_dict = {k: clean_mediawiki_markup(v) for k, v in infobox_dict.items()}

filename = cleaned_dict.get("国旗画像")


def get_image_url(filename):
    S = requests.Session()
    URL = "https://www.mediawiki.org/w/api.php"

    HEADERS = {"User-Agent": "abc (hi@example.com)"}
    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": f"File:{filename}",
        "iiprop": "url",
    }

    response = S.get(url=URL, params=PARAMS, headers=HEADERS)
    data = response.json()

    pages = data.get("query", {}).get("pages", {})

    for page_id in pages:
        image_info = pages[page_id].get("imageinfo", [])
        if image_info:
            return image_info[0].get("url")
    return None


# 2. Fetch and display the URL
flag_url = get_image_url(filename)
print(f"Flag Image URL: {flag_url}")
