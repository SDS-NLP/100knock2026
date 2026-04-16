import random


def typoglycemia(sentence: str):
    shuffled = []
    words = sentence.split()
    for word in words:
        if len(word) <= 4:
            shuffled.append(word)
            continue

        middle = list(word[1: -1])
        random.shuffle(middle)

        shuffled.append(f"{word[0]}{"".join(middle)}{word[-1]}")

    return " ".join(shuffled)

if __name__ == "__main__":
    sentence = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
    print(typoglycemia(sentence))




