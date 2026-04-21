def ngram(input, n):
    return [input[i:i + n] for i in range(len(input) - n + 1)]


if __name__ == "__main__":
    s = "I am an NLPer"
    ws = s.split(' ')
    trigrams = ngram(s, 3)
    bigrams = ngram(ws, 2)

    print(f"Trigrams: {trigrams}\nBigrams: {bigrams}")
