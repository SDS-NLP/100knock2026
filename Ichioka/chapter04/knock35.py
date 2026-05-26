import spacy
from spacy import displacy

nlp = spacy.load("ja_ginza")
doc = nlp("メロスは激怒した。")

# ブラウザで表示（http://localhost:5000 が自動で開く）
displacy.serve(doc, style="dep", options={"compact": False})