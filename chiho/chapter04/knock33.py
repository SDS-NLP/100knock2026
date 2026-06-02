from pathlib import Path
import spacy

nlp = spacy.load("ja_ginza")

file_path = Path(__file__).with_name("chapter04.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

doc = nlp(text)

for token in doc:
    if token.dep_ != "ROOT":
        print(f"{token.text}\t{token.head.text}")

