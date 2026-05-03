import MeCab

file = "merosu.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()

def get_base_verbs(text):
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)

    verbs_base =[]

    while node:
        features = node.feature.split(",")
        if features[0]=="動詞":
            verbs_base.append(features[7])

        node = node.next

    return verbs_base

def main():
    verbs = get_base_verbs(text)
    print(verbs)

if __name__=="__main__":
    main()