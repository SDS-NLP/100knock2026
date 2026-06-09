from gensim.models import KeyedVectors

analogy_path = "questions-words.txt"

model_path = "/Users/caitlyn/Downloads/GoogleNews-vectors-negative300.bin.gz"

output_path = "questions-words-result.txt"

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

section = None

semantic_total = 0
semantic_correct = 0

syntactic_total = 0
syntactic_correct = 0

with open(analogy_path, "r", encoding="utf-8") as f, \
     open(output_path, "w", encoding="utf-8") as out:

    for line in f:
        line = line.strip()

        if line == "":
            continue

        if line.startswith(":"):
            section = line[2:]
            continue

        word1, word2, word3, word4 = line.split()

        try:
            predicted_word, similarity = model.most_similar(
                positive=[word2, word3],
                negative=[word1],
                topn=1
            )[0]

        except KeyError:
            predicted_word = "NOT_FOUND"
            similarity = 0.0

        print(
            section,
            word1,
            word2,
            word3,
            word4,
            predicted_word,
            similarity,
            sep="\t",
            file=out
        )

        if section in semantic_sections:
            semantic_total += 1

            if predicted_word == word4:
                semantic_correct += 1

        elif section in syntactic_sections:
            syntactic_total += 1

            if predicted_word == word4:
                syntactic_correct += 1

semantic_accuracy = semantic_correct / semantic_total
syntactic_accuracy = syntactic_correct / syntactic_total

print("semantic analogy accuracy:", semantic_accuracy)
print("syntactic analogy accuracy:", syntactic_accuracy)

# semantic analogy accuracy: 0.7308602999210734
# syntactic analogy accuracy: 0.7400468384074942