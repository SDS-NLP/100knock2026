def n_gram(n, S):
    return [S[i: i+n] for i in range(len(S) - n + 1)]

target = "I am an NLPer"
print(n_gram(3, target))
print(n_gram(2, target))