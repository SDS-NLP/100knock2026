import json
import re
from pathlib import Path

import requests

text = str(json.load(Path("uk.json").open(encoding="utf-8")))
raw_info = re.sub(
    r"'{2,5}",
    "",
    re.search(r"\{\{基礎情報.*?\|(.*?)\}\}(?:\n)*'", text, re.DOTALL).group(1),
)

raw_info = re.sub(r"\[\[.*?\|(.*?)\]\]", r"\1", raw_info)
raw_info = re.sub(r"\[\[(.*?)\]\]", r"\1", raw_info)
raw_info = re.sub(r"\*", "", raw_info)
raw_info = re.sub(r"\</*?ref(:?erences/)*\>", "", raw_info)
raw_info = re.sub(r"\<br\s*?/\>", "", raw_info)
raw_info = re.sub(r"/\>", "", raw_info)

basic_info = re.split(r"\n\|", raw_info)
template = {
    re.split(r"=", i)[0].strip(): re.split(r"=", i)[-1].strip() for i in basic_info
}

session = requests.Session()

url = "https://en.wikipedia.org/w/api.php"

params = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": f"File:{template['国旗画像']}",
    "iiprop": "url",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windoes NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

response = session.get(url=url, params=params, headers=headers)
res_json = response.json()
pages = res_json["query"]["pages"]

for k, v in pages.items():
    print(v["imageinfo"][0]["url"]) if "imageinfo" in v else None
