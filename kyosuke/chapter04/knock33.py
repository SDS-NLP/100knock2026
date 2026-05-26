#windows環境だとうまく動かないのでターミナルでwslをしてnlp100の中で実行する
import spacy
import ginza
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
for span in ginza.bunsetu_spans(doc):
    for token in span.lefts:
        print(f'{token} : {str(ginza.bunsetu_span(token))} → {str(span)}')
for span in ginza.bunsetu_phrase_spans(doc):
    for token in span.lefts:
        print(f'{token} : {str(ginza.bunsetu_phrase_span(token))} → {str(span)}')