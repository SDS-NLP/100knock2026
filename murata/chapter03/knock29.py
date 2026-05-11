import json
import requests

with open('uk_info.json', encoding='utf-8') as f:
    info = json.load(f)

flag_filename = info["国旗画像"]
print("ファイル名:", flag_filename)

url = "https://ja.wikipedia.org/w/api.php"
params = {
    "action": "query",
    "titles": f"File:{flag_filename}",
    "format": "json",
    "prop": "imageinfo",
    "iiprop": "url",
}
headers = {
    "User-Agent": "NLP100-bot/1.0 (https://github.com/k2006NL) Python/requests"
}

r = requests.get(url, params=params, headers=headers)
data = r.json()

pages = data["query"]["pages"]
page = next(iter(pages.values()))
image_url = page["imageinfo"][0]["url"]
print("URL:", image_url)