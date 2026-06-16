import spacy
from spacy import displacy

text = "メロスは激怒した。"

# モデル読み込み
nlp = spacy.load("ja_ginza")

# 解析
doc = nlp(text)

# ターミナルに表示のurlをブラウザで確認
# https://spacy.io/usage/visualizers
displacy.serve(doc, style="dep", port=8888)