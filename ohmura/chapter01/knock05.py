def n_gram(target, n):
    result = []

    for i in range(len(target) - n + 1):
        result.append(target[i:i + n])
    return result

text = "I am an NLPer"

char_trigram = n_gram(text, 3)

words = text.split()
word_bigram = n_gram(words, 2)

print(f"文字tri-gram: {char_trigram}") 

print(f"単語bi-gram: {word_bigram}")