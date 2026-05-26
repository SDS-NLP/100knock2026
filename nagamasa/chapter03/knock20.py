import json
import pandas as pd

# 20. JSONデータの読み込み

# 方法1: 標準ライブラリ
def get_article(title):
    with open("jawiki-country.json") as f:
        for line in f:
            data = json.loads(line)
            if data["title"] == title:
                return data["text"]

uk = get_article("イギリス")

# 方法2: pandas
# df = pd.read_json("jawiki-country.json", lines=True)
# uk = df[df["title"] == "イギリス"]["text"].values[0]

if __name__ == "__main__":
    print(uk)

# json.loads()はJSON文字列をPythonの辞書に変換する
# json.load()はファイルオブジェクトから直接読み込む
# 今回は1行1JSONなのでfor line in fで一行ずつjson.loads()に渡す

# 標準ライブラリの実装はイギリスが見つかった時点でreturnする早期終了
# ファイルを全部読まないのでメモリ効率が良い

# pandasのread_json()はlines=Trueで1行1JSONを読み込む
# lines=Trueがないと全体を一つのJSONとして解釈してエラーになる
# df["title"] == "イギリス"はブールインデックス
# 各要素をTrue/Falseに変換したSeriesをdf[...]に渡すとTrueの行だけ抽出される
# .values[0]でSeriesをNumPy配列に変換して最初の要素を文字列で取り出す

# pandasは全行をメモリに載せてからフィルタする
# 標準ライブラリは早期終了できるのでメモリ効率が良い
# ただしpandasはコードが簡潔で21以降の処理も一行で書ける場面が多い