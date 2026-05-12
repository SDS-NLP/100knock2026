import random

def typoglycemia(sentence):
    res = []
    words = sentence.split(" ")

    for word in words:
        target = []
        if len(word) <= 4:
            res.append(word)
        else:
            target = list(word[1:len(target) - 1])
            random.shuffle(target)
            res.append(f"{word[0]}{''.join(target)}{word[-1]}")
    
    return " ".join(res)

if __name__ == "__main__":
    sentence = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
    print(typoglycemia(sentence))