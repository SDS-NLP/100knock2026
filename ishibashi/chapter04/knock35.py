import spacy
from spacy import displacy

def get_dependency_tree(text):
    """textの係り受け機を可視化する関数"""
    config = {"components": {"compound_splitter": {"split_mode": "A"}}}
    nlp = spacy.load('ja_ginza', config=config)
    doc = nlp(text)

    return displacy.serve(doc, style='dep', options={'compact': False}, host='127.0.0.1', auto_select_port=True)

if __name__ == "__main__":
    text = "メロスは激怒した。"

    get_dependency_tree(text)