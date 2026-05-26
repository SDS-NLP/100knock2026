import requests
from knock20 import get_uk_text
from knock25 import extract_templete

def get_flag_url(flag_name):
    """テンプレートの内容を利用し、国旗画像のURLを取得する関数"""
    url = "https://ja.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "titles": "File:" + flag_name,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json" 
    }

    response = requests.get(url, params=params).json()

    pages = response["query"]["pages"]
    for page in pages.values():
        return page["imageinfo"][0]["url"]
    
if __name__ == "__main__":
    text = get_uk_text('./chapter03/jawiki-country.json.gz')
    templete = extract_templete(text)

    flag_file = templete.get("国旗画像")

    if flag_file:
        clean_flag_file = flag_file.split('|')[0]
        
        flag_url = get_flag_url(flag_file)
        print(f"国旗画像のURL: {flag_url}")