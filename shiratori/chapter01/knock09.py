import random


def randomise(txt):
    words = txt.split()
    result = []

    for word in words:
        dot = ""
        if word.endswith("."):
            word = word[:-1]
            dot = "."

        if len(word) <= 4:
            result.append(word + dot)
        else:
            first = word[0]
            last = word[-1]
            middle = word[1:-1]
            shuffled = "".join(random.sample(middle, len(middle)))
            result.append(first + shuffled + last + dot)

    return " ".join(result)


def main():

    txt = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind."
    print(randomise(txt))


main()
