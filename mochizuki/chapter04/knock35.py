import spacy
from spacy import displacy

nlp = spacy.load('ja_ginza')
doc = nlp('メロスは激怒した。')

if __name__ == "__main__":
    svg = displacy.render(doc, style='dep', jupyter=False)
    with open('knock35.html', 'w', encoding='utf-8') as f:
        f.write(svg)
    print('knock35.html に保存しました')
