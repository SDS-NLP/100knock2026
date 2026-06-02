import spacy
from spacy import displacy
nlp = spacy.load("ja_core_news_sm")

text = "メロスは激怒した。"

doc = nlp(text)
html = displacy.render(doc, style="dep", page=True)

output_file = "dependency_tree.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print(output_file)