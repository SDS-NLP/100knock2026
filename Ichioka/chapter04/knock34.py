import spacy

nlp = spacy.load("ja_ginza")

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

doc = nlp(text)

print("述語")
print("-" * 20)

for token in doc:
    # 「メロス」が主語（nsubj）として係っているトークンを検索
    if token.dep_ == "nsubj" and token.text == "メロス":
        predicate = token.head  # 係り先が述語
        print(predicate.text)