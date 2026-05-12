import spacy

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

text = text.replace("\n", "").replace("　", "")
nlp = spacy.load("ja_ginza",config=config)
doc = nlp(text)

target_word = "メロス"

for token in doc:
    if token.text == target_word:
        predicate = token.head

        if token.dep_ == "ROOT":
            continue
    
        relation = token.dep_
        print(f"{target_word} --({relation})--> {predicate.text}")