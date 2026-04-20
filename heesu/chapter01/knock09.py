from random import shuffle
string = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
words = string.split(' ')
new_words = []
for word in words:
    if len(word) > 4:
        shuffled = list(word)
        shuffle(shuffled)
        new_words.append("".join(shuffled))
    else:
        new_words.append(word)
print(" ".join(new_words))
