import spacy

file = "merosu.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()
config = {
    "components": {
        "compound_splitter": {
            "split_mode": "A", # 分割モード「A（短い単語に分割）」を指定
        }
    }
}

text = text.replace("\n", "").replace("　", "")
nlp = spacy.load("ja_ginza",config=config)
doc = nlp(text)

for token in doc:
    print(f"{token.text} --({token.dep_})--> {token.head.text}")