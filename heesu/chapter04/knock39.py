import gzip
import json
import re
import MeCab
import matplotlib.pyplot as plt
from collections import Counter


def remove_wiki_markup(text):
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*?\|)?([^|\]]*?)\]\]", r"\1", text)
    text = re.sub(r"\[http(?:s)?://[^\s]+?\s([^\]]+?)\]", r"\1", text)
    text = re.sub(r"=+([^=]+)=+", r"\1", text)
    text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    return text


def plot_zipfs_law(file_path):
    word_counter = Counter()
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
                    if pos not in ["記号", "空白"]:
                        word_counter[node.surface] += 1
                node = node.next

    counts = [count for word, count in word_counter.most_common()]

    ranks = list(range(1, len(counts) + 1))

    plt.figure(figsize=(10, 6))

    plt.scatter(ranks, counts, s=10, alpha=0.5, color="blue")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Rank")
    plt.ylabel("Frequency")

    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    file_path = "jawiki-country.json.gz"

    plot_zipfs_law(file_path)
