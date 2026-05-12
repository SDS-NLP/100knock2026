import requests
import re 
path = 'uk.txt'
dict = {}
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
info = re.search(r'^\{\{基礎情報 国\n(.*?)\n\}\}', text, re.MULTILINE | re.DOTALL)
if info:
        info_text = info.group(1)
info_text = re.sub(r"'{2,5}(.*?)'{2,5}", r'\1', info_text)
info_text = re.sub(r"\[\[(?:[^|\]]+\|)?(.*?)\]\]", r'\1', info_text)
info_text = re.sub(r"<.*?>", "", info_text)
info_text = re.sub(r"\{\{lang\|.*?\|(.*?)\}\}", r'\1', info_text)
info_text = re.sub(r"\{\{.*?\}\}", "", info_text)
info_text = re.sub(r"\[https?://www\..*?\]", "", info_text)
matches = re.finditer(r'^\|(.+?)\s*=\s*(.*?)(?=\n\||\n\}\})', info_text, re.MULTILINE | re.DOTALL)
for match in matches:
    key = match.group(1)
    value = match.group(2)
    dict[key] = value

# (※ここはご自身の info_dict["国旗画像"] から取得してください)
flag_filename = dict["国旗画像"]

url = "https://ja.wikipedia.org/w/api.php"
headers = {
    "User-Agent": "NLP100"
}

params = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": f"File:{flag_filename}",
    "iiprop": "url"
}

# headers=headers を追加して送信！
response = requests.get(url=url, params=params, headers=headers)

data = response.json()

pages = data["query"]["pages"]
page_info = list(pages.values())[0]
flag_url = page_info["imageinfo"][0]["url"]
print(f" 国旗のURL: {flag_url}")