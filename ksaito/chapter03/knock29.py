# knock29.py
import requests
from knock28 import extract_basic_info_clean


def get_flag_url(flag_filename):
    url = "https://ja.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url",
        "titles": f"File:{flag_filename}",
    }
    headers = {
        "User-Agent": "nlp100-exercise/1.0"
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    return page["imageinfo"][0]["url"]


if __name__ == "__main__":
    input_file = "data/uk.txt"
    info = extract_basic_info_clean(input_file)
    flag_filename = info["国旗画像"]
    print("ファイル名:", flag_filename)
    print("URL:", get_flag_url(flag_filename))
