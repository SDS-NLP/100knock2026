import spacy
from spacy import displacy

config = {"components": {"compound_splitter": {"split_mode": "A"}}}
nlp = spacy.load("ja_ginza", config=config)

text = "メロスは激怒した。"
doc = nlp(text)

displacy.serve(doc, style="dep", host="127.0.0.1", port=5000)
