def n_gram(s, window: int) -> list:
    return [s[i: i + window] for i in range(0, len(s) - window + 1)]

def main():
    sentence = "I am an NLPer"
    bi = 2
    tri = 3
    print(f"bi-gram:\n{n_gram(sentence, bi)}\n")
    print(f"tri-gram:\n{n_gram(sentence, tri)}")

if __name__ == "__main__":
    main()



