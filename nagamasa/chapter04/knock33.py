import spacy
import ginza

# 33. 係り受け解析

nlp = spacy.load("ja_ginza", exclude=["compound_splitter"])
text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""
doc = nlp(text)

print("係り元\t係り先")
for span in ginza.bunsetu_spans(doc):
    parent_token = span.root.head
    if span.root != parent_token:
        parent_span = ginza.bunsetu_span(parent_token)
        print(f"{span.text}\t{parent_span.text}")









# ginza.bunsetu_spans(doc)はdocを文節単位のSpanに区切る
# 文節とは「メロスは」「激怒した」のような自然な区切り単位

# span.rootは文節の主辞トークン
# span.root.headは係り先のトークン（トークン単位）
# ginza.bunsetu_span(token)でトークンが属する文節Spanを取得する

# span.root == parent_tokenの場合は自己ループ（ルートノード）
# 係り先が自分自身なので出力しない

# spacyの依存関係解析はトークン単位で行われる
# ginzaはそれを文節単位に変換するラッパーを提供している