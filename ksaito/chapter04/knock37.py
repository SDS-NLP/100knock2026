from collections import Counter

def count_nouns(morphs):
    return Counter(m["surface"] for m in morphs if m["pos"] == "名詞")

if __name__ == "__main__":
    from morph_utils import load_text, parse_to_morphs

    text = load_text("data/kokoro.txt")
    morphs = parse_to_morphs(text)

    counter = count_nouns(morphs)

    for word, freq in counter.most_common(20):
        print(f"{word}\t{freq}")
