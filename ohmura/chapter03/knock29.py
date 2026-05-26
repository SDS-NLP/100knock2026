import gzip
import json
import re
import urllib.request
import urllib.parse

def solve_knock29_urllib():
    filename = "jawiki-country.json.gz"
    uk_text = ""
    
    # 1. ファイル読み込み
    try:
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["title"] == "イギリス":
                    uk_text = data["text"]
                    break
    except FileNotFoundError:
        print("ファイルが見つかりません。")
        return

    # 2. ファイル名の抽出
    flag_match = re.search(r"\|国旗画像\s*=\s*(.+?)\n", uk_text)
    if not flag_match:
        print("国旗画像フィールドが見つかりません。")
        return
    flag_file = flag_match.group(1).replace(" ", "_")

    # 3. urllibを使ったAPIリクエスト
    url = "https://www.mediawiki.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": f"File:{flag_file}",
        "iiprop": "url"
    }
    
    # URLの組み立て
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    
    # リクエスト（User-Agentを設定）
    req = urllib.request.Request(full_url, headers={"User-Agent": "MyNLPBot/1.0"})
    
    try:
        with urllib.request.urlopen(req) as res:
            response_data = json.loads(res.read().decode())
            pages = response_data.get("query", {}).get("pages", {})
            # 取得した辞書の最初の要素からURLを取り出す
            for v in pages.values():
                print(v.get("imageinfo", [{}])[0].get("url", "URLが見つかりません"))
    except Exception as e:
        print(f"通信エラー: {e}")

if __name__ == "__main__":
    solve_knock29_urllib()