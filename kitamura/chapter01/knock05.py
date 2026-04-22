def n_gram(text, n):
    answer = []
    for i in range(len(text)-n+1):
        answer.append(text[i:i+n])

    return answer

text = "I am an NLPer"
word = text.split()

character_tri_gram = n_gram(text, 3)
word_bi_gram = n_gram(word, 2)

print(character_tri_gram)
print(word_bi_gram)