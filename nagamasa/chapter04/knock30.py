import spacy

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

# 4. トークン（形態素）ごとの情報を取り出す
for token in doc:
    if token.tag_.startswith("動詞"):
        print(f"表層形: {token.text}")
        print(f"品詞(詳細): {token.tag_}") # 辞書（Sudachi）に基づく詳細な品詞タグ
        print("---")