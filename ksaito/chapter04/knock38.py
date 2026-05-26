# TF (Term Frequency)
# TF(t, d) = 文書 d における単語 t の出現回数 / 文書 d の総単語数
# IDF (Inverse Document Frequency)
# IDF(t) = log (総文書数 N / 単語 t を含む文書数 df)
#
# TF-IDF(t, d) = TF(t, d) * IDF(t)
import math
import re
from collections import Counter
from morph_utils import parse_to_morphs

def split_into_chapters(path):
    chapter_pattern = re.compile(r"^[一二三四五六七八九十百]+$")

    chapters = []
    current = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if chapter_pattern.match(stripped):
                if current:
                    chapters.append("".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            chapters.append("".join(current))
    return chapters

def extract_nouns(text):
    return [m["surface"] for m in parse_to_morphs(text) if m["pos"] == "名詞"]

def calc_idf(chapters_nouns):
    N = len(chapters_nouns)
    df = Counter()
    for nouns in chapters_nouns:
        for word in set(nouns):
            df[word] += 1
    return {word: math.log(N / count) for word, count in df.items()}

def calc_tf(nouns):
    counter = Counter(nouns)
    total = sum(counter.values())
    return {word: count / total for word, count in counter.items()}

if __name__ == "__main__":
    chapters = split_into_chapters("data/kokoro.txt")
    chapters_nouns = [extract_nouns(ch) for ch in chapters]
    print(f"総章数: {len(chapters)}")

    idf = calc_idf(chapters_nouns)

    target_idx = 90
    target_nouns = chapters_nouns[target_idx]
    tf = calc_tf(target_nouns)

    tfidf = {word: tf[word] * idf[word] for word in tf}

    print(f"第{target_idx}章の TF-IDF 上位20:")
    print("単語\tTF\tIDF\tTF-IDF")
    print("-" * 38)
    sorted_tfidf = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)
    for word, score in sorted_tfidf[:20]:
        print(f"{word}\t{tf[word]:.4f}\t{idf[word]:.2f}\t{score:.4f}")
