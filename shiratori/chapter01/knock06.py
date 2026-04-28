def bigram_set(seq, n=2):
    result = []
    for i in range(len(seq) - n + 1):
        result.append(seq[i : i + n])
    return set(result)


# リストだと時間がかかる


def main():
    X = bigram_set("paraparaparadise", 2)
    Y = bigram_set("paragraph", 2)

    union = X | Y
    inter = X & Y
    diff = X - Y

    print(f"和集合:{union}")
    print(f"積集合:{inter}")
    print(f"差集合:{diff}")

    print("se in X:", "se" in X)
    print("se in Y:", "se" in Y)


if __name__ == "__main__":
    main()
