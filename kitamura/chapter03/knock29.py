import re
import requests
file = "uk_article.txt"

with open(file, "r", encoding="utf-8") as f:
    content = f.read()

pattern1 = r"{{基礎情報 国\n(.*)\n}}"
match = re.search(pattern1, content, re.DOTALL) 

if match:
    basic_info = match.group(1)

pattern2 = r"^\|(.*?)\s*=\s*(.*?)(?=\n^\||\n^}|$)"
field_names = re.findall(pattern2, basic_info, re.DOTALL | re.MULTILINE)

result = {}
for field_name, v in field_names:
    result[field_name] = v

pattern_emp = r"'{2,5}"
pattern_link = r"\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]"
pattern_ref = r"\<ref.*?\>.*?\<\/ref\>|\<ref.*?\/\>"
pattern_br = r"\<br \/\>"

dict_28 = {}
for key, value in result.items():
    without_emp_value = re.sub(pattern_emp, "", value)
    without_link_value = re.sub(pattern_link, r"\1", without_emp_value)
    without_ref_value = re.sub(pattern_ref, "", without_link_value, flags=re.DOTALL)
    without_br_value = re.sub(pattern_br, "", without_ref_value, )
    dict_28[key] = without_br_value


filename = dict_28["国旗画像"]

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": f"File:{filename}",
    "iiprop": "url"
}

HEADERS = {
    "User-Agent": "MyBot/1.0 (your_email@example.com)" 
}

URL = "https://ja.wikipedia.org/w/api.php"
S = requests.Session()

R = S.get(url=URL, params=PARAMS,headers=HEADERS)

if R.status_code != 200:
    print(f"Error: {R.status_code}")

else:
    DATA = R.json()

    page_dict = DATA["query"]["pages"]

    for k, v in page_dict.items():
        if "imageinfo" in v:
            image_url = v["imageinfo"][0]["url"]
            print(image_url)
        else:
            print(f"ファイルが見つかりませんでした: {v.get('title')}")

#https://upload.wikimedia.org/wikipedia/commons/8/83/Flag_of_the_United_Kingdom_%283-5%29.svg