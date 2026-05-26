import gzip
import json
import re
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


def get_top_20_nouns(file_path):
    noun_counter = Counter()
    tagger = MeCab.Tagger()

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            text = article.get("text", "")

            cleaned_text = remove_wiki_markup(text)

            node = tagger.parseToNode(cleaned_text)

            while node:
                if node.surface:
                    features = node.feature.split(",")
                    pos = features[0]

                    if pos == "名詞":
                        noun_counter[node.surface] += 1

                node = node.next

    return noun_counter.most_common(20)


if __name__ == "__main__":
    file_path = "jawiki-country.json.gz"

    top_20_nouns = get_top_20_nouns(file_path)

    for rank, (word, count) in enumerate(top_20_nouns, 1):
        print(f"{rank:<4} | {word:<10} | {count}")
