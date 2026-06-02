results_path = "tmp/questions-words-results.txt"

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

semantic_total = 0
semantic_correct = 0

syntactic_total = 0
syntactic_correct = 0

with open(results_path, "r", encoding="utf-8") as f:
    header = next(f)

    for line in f:
        line = line.strip()

        if not line:
            continue

        cols = line.split()

        section = cols[0]
        word1 = cols[1]
        word2 = cols[2]
        word3 = cols[3]
        correct_word = cols[4]
        predicted_word = cols[5]

        # ERROR は評価対象から除外
        if predicted_word == "ERROR":
            continue

        if section in semantic_sections:
            semantic_total += 1
            if predicted_word == correct_word:
                semantic_correct += 1

        elif section in syntactic_sections:
            syntactic_total += 1
            if predicted_word == correct_word:
                syntactic_correct += 1

semantic_accuracy = semantic_correct / semantic_total
syntactic_accuracy = syntactic_correct / syntactic_total

print(f"semantic analogy accuracy: {semantic_accuracy:.6f}")
print(f"syntactic analogy accuracy: {syntactic_accuracy:.6f}")

print(f"semantic: {semantic_correct} / {semantic_total}")
print(f"syntactic: {syntactic_correct} / {syntactic_total}")