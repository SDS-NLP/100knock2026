import os
import spacy
from spacy import displacy

os.makedirs("output", exist_ok=True)

nlp = spacy.load("ja_ginza")
doc = nlp("メロスは激怒した")

svg = displacy.render(doc, style="dep")
output_path = "output/knock35.svg"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(svg)

print("output/knock35.svg に保存されました。")
