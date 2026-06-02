import spacy
import ginza

# 34. 主述の関係

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

for span in ginza.bunsetu_spans(doc):
    if span.text.strip().startswith("メロス"):
        parent_token = span.root.head
        parent_span = ginza.bunsetu_span(parent_token)
        print(f"主語：{span.text.strip()}->述語：{parent_span.text}")











# span.text.strip()で文節の前後の改行や空白を除去する
# startswith("メロス")でメロスが主語の文節を抽出する

# 係り受け解析で取れるのは明示的な主語のみ
# 「けれども邪悪に対しては、人一倍に敏感であった」のような
# 省略された主語は推定できない。これは係り受け解析の限界だ

# span.root != parent_tokenでルートノードを除外する
# ルートノードは係り先が自分自身なので出力しない