import random

words = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."


def typoglycemia(words):
    return " ".join(
        [
            (word[0] + "".join(random.sample(word[1:-1], len(word) - 2)) + word[-1])
            if len(word) > 4
            else word
            for word in words.split()
        ]
    )


print(typoglycemia(words))
