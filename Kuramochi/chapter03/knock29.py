import requests
from knock20 import extract_uk_text
from knock25 import extract_infobox

UK_TEXT  = extract_uk_text()
infobox  = extract_infobox(UK_TEXT)

print(infobox["国旗画像"])

file_name = infobox['国旗画像'].strip()
file_name = file_name.replace(' ', '_')
url       = "https://ja.wikipedia.org/w/api.php"

params = {
    "action": "query",
    "format": "json",
    "prop"  : "imageinfo",
    "titles": f"File:{file_name}",
    "iiprop": "url"
}

headers  = {'User-Agent': 'NLP100Knock_Bot/1.0'}
response = requests.get(url, params=params, headers=headers)
data     = response.json()
pages    = data['query']['pages']

for k, v in pages.items():
    image_url = v['imageinfo'][0]['url']
    print(image_url)