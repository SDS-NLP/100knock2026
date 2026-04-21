import random

def typoglycemia(sentence):
    words = sentence.split()
    result = []
    for word in words:
        if len(word) <= 4:
            result.append(word)
        else:
            middle = list(word[1:-1])
            random.shuffle(middle)
            result.append(word[0] + "".join(middle) + word[-1])
    return " ".join(result)

sentence = (
    "I couldn’t believe that I could actually "
    "understand what I was reading : the phenomenal "
    "power of the human mind ."
)
print(typoglycemia(sentence))