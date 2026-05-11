import json
import urllib.request
import urllib.parse
import knock25

filename = knock25.info['国旗画像']
params = urllib.parse.urlencode({
    'action': 'query',
    'titles': f'File:{filename}',
    'prop': 'imageinfo',
    'iiprop': 'url',
    'format': 'json'
})

req = urllib.request.Request(
    f'https://en.wikipedia.org/w/api.php?{params}',
    headers={'User-Agent': 'knock29/1.0'}
)
with urllib.request.urlopen(req) as res:
    data = json.load(res)

pages = data['query']['pages']
page = next(iter(pages.values()))
print(page['imageinfo'][0]['url'])
