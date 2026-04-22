def n_gram(sequence, n):
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]
text = "I am an NLPer"
print(n_gram(text, 3))
print(n_gram(text.split(), 2))