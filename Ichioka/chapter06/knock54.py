from gensim.models import KeyedVectors

model_path = "tmp/GoogleNews-vectors-negative300.bin.gz"
questions_path = "tmp/questions-words.txt"
output_path = "tmp/questions-words-results.txt"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

semantic_sections = {
    "capital-common-countries",
    "capital-world",
    "currency",
    "city-in-state",
    "family",
}

syntactic_sections = {
    "gram1-adjective-to-adverb",
    "gram2-opposite",
    "gram3-comparative",
    "gram4-superlative",
    "gram5-present-participle",
    "gram6-nationality-adjective",
    "gram7-past-tense",
    "gram8-plural",
    "gram9-plural-verbs",
}

current_section = None

with open(questions_path, "r", encoding="utf-8") as f_in, \
     open(output_path, "w", encoding="utf-8") as f_out:

    f_out.write("section word1 word2 word3 word4 predicted similarity\n")

    for line in f_in:
        line = line.strip()

        if not line:
            continue

        if line.startswith(":"):
            current_section = line[2:]
            continue

        words = line.split()

        if len(words) != 4:
            continue

        word1, word2, word3, word4 = words

        try:
            results = model.most_similar(
                positive=[word2, word3],
                negative=[word1],
                topn=1
            )

            predicted_word, similarity = results[0]

            f_out.write(
                f"{current_section} "
                f"{word1} {word2} {word3} {word4} "
                f"{predicted_word} {similarity:.6f}\n"
            )

        except KeyError:
            # モデルに存在しない単語がある場合はスキップ扱いしやすいように記録
            f_out.write(
                f"{current_section} "
                f"{word1} {word2} {word3} {word4} "
                f"ERROR 0.000000\n"
            )

print(f"結果を -> {output_path} ")