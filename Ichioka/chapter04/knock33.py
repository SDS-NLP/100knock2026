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

print("係り元\t係り先\t依存関係ラベル")
print("-" * 40)

for token in doc:
    if token.dep_ != "ROOT" and token.text.strip():
        print(f"{token.text}\t{token.head.text}\t{token.dep_}")