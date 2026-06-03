import re
import MeCab

def clean_text(text):
    text = re.sub(r"《[^》]*》", "", text)
    text = re.sub(r"［＃[^］]*］", "", text)
    text = text.replace("｜", "")
    return text

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return clean_text(text)

def parse_to_morphs(text):
    tagger = MeCab.Tagger()
    morphs = []
    for line in tagger.parse(text).splitlines():
        if line == "" or line == "EOS":
            continue

        surface, feature = line.split()
        f = feature.split(',')
        morphs. append({
            "surface": surface,
            "base": f[6] if len(f) > 6 else surface,
            "pos": f[0],
            "pos1": f[1]
        })
    return morphs

if __name__ == "__main__":
    text = load_text("data/kokoro.txt")
    # print(text[:200])

    morphs = parse_to_morphs(text)
    print(f"携帯その総数: {len(morphs)}")
    for m in morphs[:15]:
        print(m)

