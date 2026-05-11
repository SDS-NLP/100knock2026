import re
import requests

with open('output_knock20.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 基礎情報テンプレート抽出
template_match = re.search(r'\{\{基礎情報.*?\n(.*?)\n\}\}', text, re.DOTALL)

flag_filename = None

if template_match:
    template_text = template_match.group(1)

    # 国旗画像のファイル名抽出
    flag_match = re.search(r'^\|国旗画像\s*=\s*(.+)', template_text, re.MULTILINE)
    if flag_match:
        flag_filename = flag_match.group(1).strip()

print(f'ファイル名: {flag_filename}')

# APIで画像URL取得
if flag_filename:
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url",
        "titles": f"File:{flag_filename}",
    }

    headers = {
        "User-Agent": "my-python-script/1.0"
    } # なんかuser agents入れないと失敗した

    response = requests.get(url, params=params, headers=headers)

    data = response.json()

    pages = data["query"]["pages"]

    for page in pages.values():
        if "imageinfo" in page:
            image_url = page["imageinfo"][0]["url"]
            print(f"国旗画像URL: {image_url}")
        else:
            print("画像が取得できませんでした")