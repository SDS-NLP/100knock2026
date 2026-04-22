def get_ngram(seq, n):
    result = []
    for i in range(len(seq) - n + 1):
        result.append(seq[i : i + n])
    return result


def main():
    txt = "I am an NLPer"
    char_tri = get_ngram(txt, 3)
    print(char_tri)

    words = txt.split()
    word_bi = get_ngram(words, 2)
    print(word_bi)


main()
