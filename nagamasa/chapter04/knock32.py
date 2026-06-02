import spacy
from spacy.matcher import Matcher

# 1. 日本語モデル（GiNZA）のロード
nlp = spacy.load("ja_ginza", exclude=["compound_splitter"])

# 2. 解析対象のテキスト
text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

# 3. テキストを解析し、Docオブジェクトを生成
doc = nlp(text)

# Matcherの初期化
matcher = Matcher(nlp.vocab)

# 「名詞 + 助詞の『の』 + 名詞」というパターンを定義
# UDタグ（POS）を用いて、世界共通の品詞分類でシンプルに記述するのがモダンな手法
pattern = [
    {"POS": "NOUN"}, 
    {"TEXT": "の", "POS": "ADP"}, 
    {"POS": "NOUN"}
]
matcher.add("NOUN_NO_NOUN", [pattern])

# Docオブジェクトに対してマッチングを実行
matches = matcher(doc)

# 結果の出力
for match_id, start, end in matches:
    matched_span = doc[start:end] # マッチした範囲をスライスで取得
    print(f"名詞句: {matched_span.text}")
    print("---")