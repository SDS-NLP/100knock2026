from knock20 import get_article
from knock28 import clean
import re
import requests

# 29. 国旗画像のURLを取得する

uk = get_article("イギリス")
block = re.search(r'{{基礎情報.+?\n}}', uk, re.DOTALL)
matches = re.findall(r'\|(.+?)\s*=\s*([\s\S]+?)(?=\n\||\n}})', block.group(0))
info = {field.strip(): clean(value) for field, value in matches}

file = info['国旗画像'].strip()
params = {
    'action': 'query',
    'titles': f'File:{file}',
    'prop': 'imageinfo',
    'iiprop': 'url',
    'format': 'json'
}
headers = {'User-Agent': 'NLP100knock/1.0 (your-email@example.com)'}
response = requests.get('https://en.wikipedia.org/w/api.php', params=params, headers=headers)
data = response.json()

page = next(iter(data['query']['pages'].values()))
print(page['imageinfo'][0]['url'])

# requestsライブラリでHTTPリクエストを送る
# requests.get(url, params=params, headers=headers)
# paramsを辞書で渡すとrequestsが自動でURLエンコードする
# スペースは%20に変換される

# WikipediaのAPIはUser-Agentの設定を要求する
# 設定しないと403が返る
# headers={'User-Agent': '...'}で設定する

# APIレスポンスの構造
# data['query']['pages']はページIDをキーとした辞書
# ページIDは動的に変わるのでvalues()で取り出す
# next(iter(...))で最初の要素を取り出す
# list(...)[0]でも同じだがメモリ効率はnext(iter(...))が良い

# imageinfo[0]['url']が画像のURL
# descriptionurlはWikipediaのファイルページのURL
# descriptionshortURLは短縮URL