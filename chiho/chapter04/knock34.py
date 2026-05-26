from pathlib import Path
import spacy

nlp = spacy.load("ja_ginza")

file_path = Path(__file__).with_name("chapter04.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

doc = nlp(text)


def predicate_text(sent):
    root = next(token for token in sent if token.dep_ == "ROOT")

    if root.pos_ == "NOUN":
        cop_heads = [token for token in root.children if token.dep_ == "cop"]
        cop_tokens = cop_heads[:]
        for cop_head in cop_heads:
            cop_tokens.extend(token for token in sent if token.head == cop_head and token.dep_ == "fixed")
        cop_tokens = sorted(cop_tokens, key=lambda token: token.i)
        if cop_tokens:
            return "".join(token.text for token in cop_tokens)

    predicate_tokens = sorted(
        [root] + [token for token in root.children if token.dep_ in {"aux", "fixed"}],
        key=lambda token: token.i,
    )
    return "".join(token.text for token in predicate_tokens)


for sent in doc.sents:
    if any(token.text == "メロス" and token.dep_ in {"nsubj", "obl"} for token in sent):
        print(predicate_text(sent))
