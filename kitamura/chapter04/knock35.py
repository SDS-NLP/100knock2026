import spacy
from spacy import displacy

file = "merosu.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()
config = {
    "components": {
        "compound_splitter": {
            "split_mode": "A", 
        }
    }
}

nlp = spacy.load("ja_ginza", config=config)
doc = nlp(text)

displacy.serve(doc, style="dep",auto_select_port=True)

# http://localhost:5001