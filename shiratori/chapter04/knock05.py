import spacy
from spacy import displacy


def visualize_dependency(text: str):
    nlp = spacy.load("ja_ginza")

    doc = nlp(text)

    displacy.serve(doc, style="dep", auto_select_port=True)
    # dep→文のどの単語がどの単語にかかっているか


if __name__ == "__main__":
    visualize_dependency("メロスは激怒した。")
