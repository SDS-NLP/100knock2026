def n_gram(target, n):
  return [target[idx:idx + n] for idx in range(len(target) - n + 1)]

sentence = "I am an NLPer"
tri_gram = n_gram(sentence, 3)
bi_gram = n_gram(sentence.split(), 2)

print(f"tri_gram：{tri_gram}")
print(f"bi_gram：{bi_gram}")