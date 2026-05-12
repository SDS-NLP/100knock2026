# 29. 国旗画像のURLを取得する
# テンプレートの内容を利用し、国旗画像のURLを取得せよ。
# （ヒント: MediaWiki APIのimageinfoを呼び出して、ファイル参照をURLに変換すればよい）

import re
import json
import urllib.parse
import urllib.request

with open("/Users/caitlyn/Documents/uk.txt", "r", encoding="utf-8") as f:
    text = f.read()

template = re.search(r"\{\{基礎情報.*?\n(.*?)\n\}\}", text, re.DOTALL).group(1)
pattern = r"^\|(.*?)\s*=\s*(.*?)(?=\n\|.+?\s*=|\Z)"

basic_info = {}

for match in re.finditer(pattern, template, re.DOTALL | re.MULTILINE):
    field_name = match.group(1).strip()
    value = match.group(2).strip()
    basic_info[field_name] = value

filename = basic_info["国旗画像"]

params = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "File:" + filename,
    "iiprop": "url",
}

url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)

req = urllib.request.Request(
    url,
    headers={"User-Agent": "100knock-study/1.0"}
)

with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode("utf-8"))

pages = data["query"]["pages"]
page = next(iter(pages.values()))

print(page["imageinfo"][0]["url"])