def generate_ngram(sequence, n, by_characters=False):
    processed = sequence if by_characters else sequence.split()
    return [processed[i : i + n] for i in range(len(processed) - (n - 1))]


if __name__ == "__main__":
    test = "I am an NLPer"
    print(f"character-gram:\n{generate_ngram(test, 3, True)}")
    print(f"word-gram:\n{generate_ngram(test, 2)}")
