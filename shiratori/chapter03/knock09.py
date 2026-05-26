from knock08 import cleaned3_info

import requests


file_name = cleaned3_info["国旗画像"]
print(file_name)

url = "https://ja.wikipedia.org/w/api.php"

params = {"action": "query", "format": "json", "prop": "imageinfo", "titles": f"File:{file_name}", "iiprop": "url"}

headers = {"User-Agent": "my-nlp100knock-script/1.0"}

res = requests.get(url, params=params, headers=headers)

data = res.json()

for _, page in data["query"]["pages"].items():
    print(page["imageinfo"][0]["url"])
