import MeCab

file = "merosu.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()

def get_verbs(text):
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)

    verbs = []

    while node:
        features = node.feature.split(",")
        if features[0] == "動詞":
            verbs.append(node.surface)
        
        node = node.next

    return verbs

def main():
    verbs = get_verbs(text)
    print(verbs)

if __name__ == "__main__":
    main()