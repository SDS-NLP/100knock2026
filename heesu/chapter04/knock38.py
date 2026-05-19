import gzip
import json
import re
import math
import MeCab
from collections import Counter


def remove_wiki_markup(text):
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*?\|)?([^|\]]*?)\]\]", r"\1", text)
    text = re.sub(r"\[http(?:s)?://[^\s]+?\s([^\]]+?)\]", r"\1", text)
    text = re.sub(r"=+([^=]+)=+", r"\1", text)
    text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    return text


def calculate_tfidf_for_japan(file_path):
    tagger = MeCab.Tagger()

    total_docs = 0
    df_counter = Counter()
    japan_tf = Counter()

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            title = article.get("title", "")
            text = article.get("text", "")

            total_docs += 1
            cleaned_text = remove_wiki_markup(text)

            nouns_in_this_doc = set()

            node = tagger.parseToNode(cleaned_text)
            while node:
                if node.surface:
                    features = node.feature.split(",")
                    if features[0] == "名詞":
                        word = node.surface
                        nouns_in_this_doc.add(word)

                        if title == "日本":
                            japan_tf[word] += 1
                node = node.next

            for word in nouns_in_this_doc:
                df_counter[word] += 1

    tfidf_scores = []

    for word, tf in japan_tf.items():
        df = df_counter[word]
        idf = math.log(total_docs / df)
        tfidf = tf * idf

        tfidf_scores.append({"word": word, "tf": tf, "idf": idf, "tfidf": tfidf})

    tfidf_scores.sort(key=lambda x: x["tfidf"], reverse=True)

    return tfidf_scores[:20]


if __name__ == "__main__":
    file_path = "jawiki-country.json.gz"

    top_20_tfidf = calculate_tfidf_for_japan(file_path)

    for rank, item in enumerate(top_20_tfidf, 1):
        print(
            f"{rank:<4} | {item['word']:<12} | {item['tf']:<10} | {item['idf']:<14.4f} | {item['tfidf']:.4f}"
        )
